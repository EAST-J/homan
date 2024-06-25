# Copyright (c) Facebook, Inc. and its affiliates.
"""
Utilities for computing initial object pose fits from instance masks.
"""
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation
# pylint: disable=no-member,import-error,abstract-method,missing-function-docstring,invalid-name,too-many-locals
import os

import matplotlib.pyplot as plt
import neural_renderer as nr
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import cv2

from libyana.camutils import project
from libyana.conversions import npt
from libyana.lib3d import kcrop
from libyana.metrics import iou as ioumetrics
from libyana.visutils import imagify
from libyana.verify import checkshape

from homan.constants import REND_SIZE
from homan.lib3d.optitrans import (
    TCO_init_from_boxes_zup_autodepth,
    compute_optimal_translation,
)
from homan.utils.nmr_renderer import PerspectiveRenderer
from homan.utils.geometry import (
    compute_random_rotations,
    rot6d_to_matrix,
    matrix_to_rot6d,
)


class PoseOptimizer(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh.
    Closely following PHOSA, we optimize an occlusion-aware silhouette loss
    that consists of a one-way chamfer loss and a silhouette matching loss.
    """
    def __init__(
        self,
        ref_image,
        depth,
        vertices,
        faces,
        textures,
        rotation_init,
        translation_init,
        num_initializations=1,
        kernel_size=7,
        K=None,
        power=0.25,
        lw_chamfer=0,
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super().__init__()

        self.register_buffer("vertices",
                             vertices.repeat(num_initializations, 1, 1))
        self.register_buffer("faces", faces.repeat(num_initializations, 1, 1))
        self.register_buffer(
            "textures", textures.repeat(num_initializations, 1, 1, 1, 1, 1))
        # Load reference mask.
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        image_ref = torch.from_numpy((ref_image > 0).astype(np.float32)) # 物体的分割部分
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32)) # 除了被遮挡的部分，即用来计算loss的部分
        depth = torch.from_numpy(depth.astype(np.float32))
        self.register_buffer("image_ref",
                             image_ref.repeat(num_initializations, 1, 1))
        self.register_buffer("keep_mask",
                             keep_mask.repeat(num_initializations, 1, 1))
        self.register_buffer("depth", 
                             depth[None, :, :].repeat(num_initializations, 1, 1))
        # self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
        #                                stride=1,
        #                                padding=(kernel_size // 2))
        self.rotations = nn.Parameter(rotation_init.clone().float(),
                                      requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(num_initializations, 1,
                                                       1)
        self.translations = nn.Parameter(translation_init.clone().float(),
                                         requires_grad=True)
        # Setup renderer.
        if K is None:
            K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        rot = torch.eye(3).unsqueeze(0).cuda()
        trans = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=rot,
            t=trans,
            orig_size=1,
            anti_aliasing=False,
        )
        self.lw_chamfer = lw_chamfer
        self.K = K

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = rot6d_to_matrix(self.rotations)
        return torch.matmul(self.vertices, rots) + self.translations

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means coord_xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.renderer.K,
            self.renderer.R,
            self.renderer.t,
            self.renderer.dist_coeffs,
            orig_size=1,
        )
        coord_xy, coord_z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(coord_z)
        lower_right = torch.max(coord_xy - 1,
                                zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum(dim=(1, 2))
        too_far = torch.max(coord_z - self.renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def comput_depth_loss(self, verts, faces):
        rend_depth = self.renderer(verts, faces, mode="depth") # bs * 256 * 256
        rend_sil = self.renderer(verts, faces, mode="silhouettes")
        rend_depth[torch.logical_or(rend_sil!=1, self.keep_mask!=1)] = 100
        rend_depth_min = torch.min(rend_depth.reshape(rend_depth.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1)
        rend_depth[torch.logical_or(rend_sil!=1, self.keep_mask!=1)] = -1
        rend_depth_max = torch.max(rend_depth.reshape(rend_depth.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1) # bs * 1 * 1
        # 背景值为0 (只获取物体的相对深度，以物体距离相机最远的点为基准)
        normalized_rend_depth = (rend_depth_max - rend_depth) / (rend_depth_max - rend_depth_min)
        normalized_rend_depth[torch.logical_or(rend_sil!=1, self.keep_mask!=1)] = 0
        gt_depth = self.depth.clone()
        gt_depth[self.image_ref!=1] = 1e6
        gt_depth_min = torch.min(gt_depth.reshape(rend_depth.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1)
        gt_depth[self.image_ref!=1] = -1
        gt_depth_max = torch.max(gt_depth.reshape(rend_depth.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1)
        normalized_gt_depth = (gt_depth - gt_depth_min) / (gt_depth_max - gt_depth_min)
        normalized_gt_depth[self.image_ref!=1] = 0
        depth_loss = torch.sum(
            self.image_ref * (normalized_rend_depth - normalized_gt_depth)**2, dim=(1, 2))
        return depth_loss

    def forward(self):
        verts = self.apply_transformation()
        image = self.keep_mask * self.renderer(
            verts, self.faces, mode="silhouettes") # 渲染物体3D Model所获得的mask(手部部分会被自动mask掉)
        loss_dict = {}
        loss_dict["mask"] = torch.sum((image - self.image_ref)**2, dim=(1, 2))
        # 考虑depth loss的权重
        loss_dict["depth"] = self.comput_depth_loss(verts, self.faces)
        with torch.no_grad():
            iou = ioumetrics.batch_mask_iou(image.detach(),
                                            self.image_ref.detach())
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        return loss_dict, iou, image

    def render(self):
        """
        Renders objects according to current rotation and translation.
        """
        verts = self.apply_transformation()
        images = self.renderer(verts, self.faces, torch.tanh(self.textures))[0]
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return images


def visualize_optimal_poses(model,
                            image_crop,
                            mask,
                            score=0,
                            save_path="tmpbestobj.png"):
    """
    Visualizes the 8 best-scoring object poses.

    Args:
        model (PoseOptimizer).
        image_crop (H x H x 3).
        mask (M x M x 3).
        score (float): Mask confidence score (optional).
    """
    num_vis = 8
    rotations = model.rotations
    translations = model.translations
    verts = model.vertices[0]
    faces = model.faces[0]
    loss_dict, _, _ = model()
    losses = sum(loss_dict.values())
    camintr_roi = model.renderer.K
    inds = torch.argsort(losses)[:num_vis]
    obj_renderer = PerspectiveRenderer()

    fig = plt.figure(figsize=((10, 4)))
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(image_crop)
    ax1.axis("off")
    ax1.set_title("Cropped Image")

    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(mask)
    ax2.axis("off")
    if score > 0:
        ax2.set_title(f"Mask Conf: {score:.2f}")
    else:
        ax2.set_title("Mask")

    for i, ind in enumerate(inds.cpu().numpy()):
        plt_ax = fig.add_subplot(2, 5, i + 3)
        rend = obj_renderer(
            vertices=verts,
            faces=faces,
            image=image_crop,
            translation=translations[ind],
            rotation=rot6d_to_matrix(rotations)[ind],
            color_name="red",
            K=camintr_roi,
        )
        plt_ax.imshow(rend)
        plt_ax.set_title(f"Rank {i}: {losses[ind]:.1f}")
        plt_ax.axis("off")
    plt.savefig(save_path)


def find_optimal_pose(
    vertices,
    faces,
    mask,
    bbox,
    square_bbox,
    image_size,
    depth,
    K=None,
    num_iterations=50,
    num_initializations=2000,
    lr=1e-2,
    image=None,
    debug=True,
    viz_folder="tmp",
    viz_step=10,
    sort_best=True,
    rotations_init=None,
    viz=True,
):
    """
    Mostly follows PHOSA :)
    """
    os.makedirs(viz_folder, exist_ok=True)
    ts = 1
    textures = torch.ones(faces.shape[0], ts, ts, ts, 3,
                          dtype=torch.float32).cuda()
    x, y, b, _ = square_bbox
    L = max(image_size[:2])
    camintr_roi = kcrop.get_K_crop_resize(
        torch.Tensor(K).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
        [REND_SIZE]).cuda()
    # Stuff to keep around
    best_losses = np.inf
    best_rots = None
    best_trans = None
    best_loss_single = np.inf
    loop = tqdm(total=num_iterations)
    K = npt.tensorify(K).unsqueeze(0).to(vertices.device)
    # Mask is in format 256 x 256 (REND_SIZE x REND_SIZE)
    # bbox is in xywh in original image space
    # K is in pixel space
    # If initial rotation is not provided, it is sampled
    # uniformly from SO3
    if rotations_init is None:
        rotations_init = compute_random_rotations(num_initializations,
                                                  upright=False)

    # Translation is retrieved by matching the tight bbox of projected
    # vertices with the bbox of the target mask
        
    # translations_init = compute_optimal_translation(
    #     bbox_target=np.array(bbox) * REND_SIZE / L,
    #     vertices=torch.matmul(vertices.unsqueeze(0), rotations_init),
    #     f=K[0, 0, 0].item() / max(image_size))
    translations_init = TCO_init_from_boxes_zup_autodepth(
        bbox, torch.matmul(vertices.unsqueeze(0), rotations_init),
        K).unsqueeze(1)

    # Bring crop K to NC rendering space
    camintr_roi[:, :2] = camintr_roi[:, :2] / REND_SIZE

    model = PoseOptimizer(
        ref_image=mask,
        depth=depth,
        vertices=vertices,
        faces=faces,
        textures=textures,
        rotation_init=matrix_to_rot6d(rotations_init),
        translation_init=translations_init,
        num_initializations=num_initializations,
        K=camintr_roi,
        lw_chamfer=0.0,
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss_dict, iou, sil = model()
        losses = sum(loss_dict.values())
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        ind = torch.argmin(losses)
        if debug and (step % viz_step == 0):
            debug_viz_folder = os.path.join(viz_folder, "poseoptim")
            os.makedirs(debug_viz_folder, exist_ok=True)
            if losses.min() < best_loss_single:
                best_sil = sil[ind].clone().detach().cpu().numpy()
                vis_mask = np.hstack([best_sil, mask])
                cv2.imwrite(os.path.join(debug_viz_folder,f"{step:04d}.png"), vis_mask * 255)
        best_loss_single = losses[ind]
        loop.set_description(f"loss: {best_loss_single.item():.3g}")
        loop.update()
    if best_rots is None:
        best_rots = model.rotations
        best_trans = model.translations
        best_losses = losses
    else:
        best_rots = torch.cat((best_rots, model.rotations), 0)
        best_trans = torch.cat((best_trans, model.translations), 0)
        best_losses = torch.cat((best_losses, losses))
    if sort_best:
        inds = torch.argsort(best_losses)
        best_losses = best_losses[inds][:num_initializations].detach().clone()
        best_trans = best_trans[inds][:num_initializations].detach().clone()
        best_rots = best_rots[inds][:num_initializations].detach().clone()
    loop.close()

    model.rotations = nn.Parameter(best_rots)
    model.translations = nn.Parameter(best_trans)
    return model


def find_optimal_poses(image_size,
                       faces=None,
                       vertices=None,
                       annotations=None,
                       images=None,
                       Ks=None,
                       num_iterations=50,
                       num_initializations=2000,
                       viz_path="tmp.png",
                       debug=False):
    """
    Compute initial object poses.
    Initialize num_initializations initial pose candidates using randomly sampled rotations and heuristic
    to match the object mesh *projected* bounding box to the *detected* bounding box.
    For each frame, refine the object translation and rotation by differentiable rendering
    to match the target computed object mask.
    Initialize the pose for the next frame with the previous pose.

    This results in num_initializations candidate object motions (each of a length frame_nb, as many
    as there are provided images).

    We keep the best candidate by keeping the motion which results in the lowest average IoU
    detween *rendered* and *target* masks over the sequence.

    Arguments:
        image_size (tuple): (height, width[, channels]]) image dimensions
        faces (np.ndarray): (face_nb, 3) object faces
        vertices (np.ndarray): (face_nb, 3) object vertices
        annotations (list[dict]): list of bbox and mask annotations
        images (list[np.ndarray]): list of frame_nb (height, width, 3) RGB images depicting
            hand-object interactions
        num_initializations (int): TODO
        Ks (list[np.ndarray]): List containing frame_nb (3, 3) intrinsic camera parameters
        num_iterations (int): Number of optimization steps

    Returns:
        list[dict]: List of initial poses (rotations, translations) and mask information
            (K_roi, masks, full_mask, target_masks)
    """
    # Check input shapes
    checkshape.check_shape(faces, (-1, 3), "faces")
    checkshape.check_shape(vertices, (-1, 3), "vertices")

    # Convert inputs to tensors
    vertices = npt.tensorify(vertices).cuda()
    faces = npt.tensorify(faces).cuda()

    # Keep track of previous rotations to get temporally consistent initialization
    previous_rotations = None
    all_object_parameters = []
    all_losses = []
    for image, annotation, K in zip(images, annotations, Ks):
        # Optimize pose to given mask evidence
        model = find_optimal_pose(
            vertices=vertices,
            faces=faces,
            image=image,
            mask=annotation["target_crop_mask"],
            bbox=annotation["bbox"],
            depth=annotation["depths"],
            square_bbox=annotation["square_bbox"],
            image_size=image_size,
            K=K,
            num_iterations=num_iterations,
            num_initializations=num_initializations,
            debug=debug,
            viz_folder=os.path.dirname(viz_path),
            sort_best=False,  # Keep initial ordering
            rotations_init=previous_rotations,
        )
        _, iou, _ = model()
        verts_trans = model.apply_transformation()
        object_parameters = {
            "rotations": rot6d_to_matrix(model.rotations).detach(),
            "translations": model.translations.detach(),
            "target_masks":
            torch.from_numpy(annotation["target_crop_mask"]).cuda(),
            "K_roi": model.K.detach(),
            "masks": annotation["full_mask"].cuda(),
            "verts": vertices.detach(),
            "verts_trans": verts_trans.detach(),
        }
        if "flow" in annotation.keys(): # 最后一帧没有光流的监督
            object_parameters.update({"flows": torch.from_numpy(annotation["flow"]).cuda()})
        all_object_parameters.append(object_parameters)
        previous_rotations = rot6d_to_matrix(model.rotations.detach(
        ))  # num_initializations, 3, 2 rot6d rotations
        all_losses.append(iou.detach())
    all_losses = torch.stack(all_losses)  # frame_nb, num_initializations

    # Keep best motion as the one which has lowest average loss over the sequence
    best_idx = torch.argsort(all_losses.mean(0))[-1]
    all_final_params = []
    # Aggregate object pose candidates for all frames
    for obj_params, info in zip(all_object_parameters, annotations):
        final_params = {}
        # Get best parameters
        for key in ["rotations", "translations", "verts_trans"]:
            final_params[key] = obj_params[key][best_idx].unsqueeze(0).cuda()

        # Copy useful mask information
        for key in ["target_masks", "K_roi", "masks", "verts"]:
            final_params[key] = obj_params[key].unsqueeze(0).cuda()
        if "flows" in obj_params.keys():
            final_params["flows"] = obj_params["flows"].unsqueeze(0).cuda()
        if "depths" in info.keys():
            final_params["depths"] = torch.from_numpy(info["depths"]).unsqueeze(0).cuda()
        final_params["full_mask"] = info["full_mask"].cuda()
        all_final_params.append(final_params)
    return all_final_params


def find_optimal_poses_indiviual(image_size,
                       faces=None,
                       vertices=None,
                       annotations=None,
                       images=None,
                       Ks=None,
                       num_iterations=50,
                       num_initializations=2000,
                       viz_path="tmp.png",
                       debug=False):
    """
    
    """
    # Check input shapes
    checkshape.check_shape(faces, (-1, 3), "faces")
    checkshape.check_shape(vertices, (-1, 3), "vertices")

    # Convert inputs to tensors
    vertices = npt.tensorify(vertices).cuda()
    faces = npt.tensorify(faces).cuda()

    # Keep track of previous rotations to get temporally consistent initialization
    previous_rotations = None
    all_object_parameters = []
    for image, annotation, K in zip(images, annotations, Ks):
        # Optimize pose to given mask evidence
        model = find_optimal_pose(
            vertices=vertices,
            faces=faces,
            image=image,
            mask=annotation["target_crop_mask"],
            bbox=annotation["bbox"],
            square_bbox=annotation["square_bbox"],
            depth=annotation["depths"],
            image_size=image_size,
            K=K,
            num_iterations=num_iterations,
            num_initializations=num_initializations,
            debug=debug,
            viz_folder=os.path.dirname(viz_path),
            sort_best=True,
            rotations_init=previous_rotations,
        )
        _, iou, _ = model()
        # best_idx = torch.argsort(iou)[-1]
        best_idx = 0
        verts_trans = model.apply_transformation()
        object_parameters = {
            "rotations": rot6d_to_matrix(model.rotations).detach()[best_idx].clone(),
            "translations": model.translations.detach()[best_idx].clone(),
            "target_masks":
            torch.from_numpy(annotation["target_crop_mask"]).cuda(),
            "K_roi": model.K.detach(),
            "masks": annotation["full_mask"].cuda(),
            "verts": vertices.detach(),
            "verts_trans": verts_trans.detach()[best_idx].clone(),
        }
        if "flow" in annotation.keys(): # 最后一帧没有光流的监督
            object_parameters.update({"flows": torch.from_numpy(annotation["flow"]).cuda()})
        all_object_parameters.append(object_parameters)
        previous_rotations = rot6d_to_matrix(model.rotations.detach().clone())
        # previous_rotations = rot6d_to_matrix(model.rotations.detach()[best_idx:best_idx+1].repeat(num_initializations, 1, 1))  # num_initializations, 3, 2 rot6d rotations
    all_final_params = []
    # Aggregate object pose candidates for all frames
    for obj_params, info in zip(all_object_parameters, annotations):
        final_params = {}
        # Get best parameters
        for key in ["rotations", "translations", "verts_trans"]:
            final_params[key] = obj_params[key].unsqueeze(0).cuda()

        # Copy useful mask information
        for key in ["target_masks", "K_roi", "masks", "verts"]:
            final_params[key] = obj_params[key].unsqueeze(0).cuda()
        if "flows" in obj_params.keys():
            final_params["flows"] = obj_params["flows"].unsqueeze(0).cuda()
        if "depths" in info.keys():
            final_params["depths"] = torch.from_numpy(info["depths"]).unsqueeze(0).cuda()
        final_params["full_mask"] = info["full_mask"].cuda()
        all_final_params.append(final_params)
    return all_final_params