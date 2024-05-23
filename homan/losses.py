#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member,wrong-import-order,too-many-arguments,too-many-instance-attributes
# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring
import neural_renderer as nr
import torch
import torch.nn.functional as F

from homan.constants import INTERACTION_MAPPING, INTERACTION_THRESHOLD, REND_SIZE
from homan.utils.bbox import check_overlap, compute_iou
from homan.utils.geometry import compute_dist_z

import itertools
from libyana import distutils
from libyana.camutils import project
from libyana.metrics.iou import batch_mask_iou
from scipy.ndimage.morphology import distance_transform_edt
import point_cloud_utils as pcu
import numpy as np
# from physim.datasets import collate


def project_bbox(vertices, renderer, bbox_expansion=0.0):
    """
    Computes the 2D bounding box of the vertices after projected to the image plane.

    Args:
        vertices (V x 3).
        renderer: Renderer used to get camera parameters.
        bbox_expansion (float): Amount to expand the bounding boxes.

    Returns:
        If a part_label dict is given, returns a dictionary mapping part name to bbox.
        Else, returns the projected 2D bounding box.
    """
    worldverts = (vertices * torch.Tensor([[[1, -1, 1.0]]]).cuda())
    proj = nr.projection(
        worldverts,
        K=renderer.K,
        R=renderer.R,
        t=renderer.t,
        dist_coeffs=renderer.dist_coeffs,
        orig_size=1,
    )
    proj = proj[:, :, :2]
    bboxes_xy = torch.cat([proj.min(1)[0], proj.max(1)[0]], 1)
    if bbox_expansion:
        center = (bboxes_xy[:, :2] + bboxes_xy[:, 2:]) / 2
        extent = (bboxes_xy[:, 2:] - bboxes_xy[:, :2]) / 2 * (1 +
                                                              bbox_expansion)
        bboxes_xy = torch.cat([center - extent, center + extent], 1)
    return bboxes_xy

def interpolate_barycentric_coords(f, fi, bc, attribute):
    """
    Interpolate an attribute stored at each vertex of a mesh across the faces of a triangle mesh using
    barycentric coordinates

    Args:
        f : a (#faces, 3)-shaped Tensor of mesh faces (indexing into some vertex array).
        fi: a (#attribs,)-shaped Tensor of indexes into f indicating which face each attribute lies within.
        bc: a (#attribs, 3)-shaped Tensor of barycentric coordinates for each attribute
        attribute: a (#vertices, dim)-shaped Tensor of attributes at each of the mesh vertices

    Returns:
        A (#attribs, dim)-shaped array of interpolated attributes.
    """
    return (attribute[f[fi.to(torch.long)]] * bc[:, :, None]).sum(1)

def get_rays(pixels_x, pixels_y, K_inv, c2w):
    """
    Generate the rays_o and rays_v
    pixels_x: (N,) 
    pixels_y: (N,)
    K_inv: (3 * 3) The inverse matrix of the camera
    c2w: (4 * 4) The transformation matrix from the camera coordinate to the world coordinate
    """
    pts = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    pts = torch.matmul(pts, K_inv.T) # batch_size, 3
    rays_d = pts / torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    rays_d = torch.matmul(rays_d, c2w[:3, :3].T) # batch_size, 3
    rays_o = c2w[:3, -1].expand(rays_d.shape) # batch_size, 3
    return rays_o, rays_d

def ray_mesh_intersect(obj_v, obj_f, rays_o, rays_d, transformation_matrix=None):
    # ! 由于暂时没有找到differentiable的ray/mesh intersector的方案，故基于np完成，梯度借助于坐标系转换传播
    '''
    obj_v: V * 3 (Tensor Need to be calculate the loss)
    obj_f: F * 3
    rays_o: N * 3
    rays_d: N * 3
    '''
    intersector = pcu.RayMeshIntersector(obj_v.detach().cpu().numpy(), obj_f.detach().cpu().numpy())
    fid, bc, t = intersector.intersect_rays(rays_o.detach().cpu().numpy().astype(np.float32), 
                                            rays_d.detach().cpu().numpy().astype(np.float32))
    hit_mask = np.isfinite(t)
    # The hit pose are in the camera-coordinate (need to be transform back to the object)
    hit_pos = interpolate_barycentric_coords(obj_f, torch.from_numpy(fid[hit_mask]).to(obj_v.device), 
                                            torch.from_numpy(bc[hit_mask]).to(obj_v.device), obj_v)
    if transformation_matrix is not None:
        hit_pos = hit_pos @ transformation_matrix[:3, :3].T + transformation_matrix[:3, -1:].T
    hit_mask = torch.from_numpy(hit_mask).to(hit_pos.device)
    hit_pose_full = torch.zeros((rays_o.shape[0], 3)).to(hit_pos.device)
    hit_pose_full[hit_mask] = hit_pos
    return hit_pose_full, hit_mask

class Losses_Obj():
    def __init__(
        self,
        renderer,
        ref_mask_object,
        keep_mask_object,
        full_mask_object,
        flow_object,
        camintr_rois_object,
        camintr,
        class_name,
        inter_type="min",
        hand_nb=1,
    ):
        """
        Args:
            inter_type (str): [centroid|min] centroid to penalize centroid distances
                min to penalize minimum distance
        """
        self.renderer = nr.renderer.Renderer(image_size=REND_SIZE,
                                             K=renderer.K,
                                             R=renderer.R,
                                             t=renderer.t,
                                             orig_size=1)
        self.inter_type = inter_type
        self.ref_mask_object = ref_mask_object
        self.keep_mask_object = keep_mask_object
        self.camintr_rois_object = camintr_rois_object
        self.full_mask_object = full_mask_object
        self.flow_object = flow_object # (bs-1) *  H * W
        # Necessary ! Otherwise camintr gets updated for some reason TODO check
        self.camintr = camintr.clone()
        self.thresh = 3  # z thresh for interaction loss
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name

        self.expansion = 0.2
        self.interaction_map = INTERACTION_MAPPING[class_name]
        self.pool = torch.nn.MaxPool2d(kernel_size=7,
                                       stride=1,
                                       padding=(7 // 2))
        mask_edge = self.compute_edges(ref_mask_object).detach().cpu().numpy() # 获取mask的边缘信息
        edt = distance_transform_edt(1 - (mask_edge > 0))**(0.5) # 获取所有像素点到边缘的距离
        self.edt_ref_edge = torch.from_numpy(edt).float().to(self.ref_mask_object.device)
        self.interaction_pairs = None

    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

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
                                zeros).sum()  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum()  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum()
        too_far = torch.max(coord_z - self.renderer.far, zeros).sum()
        return lower_right + upper_left + behind + too_far

    def compute_sil_loss_object(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        # Rendering happens in ROI
        camintr = self.camintr_rois_object
        rend = self.renderer(verts, faces, K=camintr, mode="silhouettes")
        image = self.keep_mask_object * rend
        l_m = torch.sum(
            (image - self.ref_mask_object)**2) / self.keep_mask_object.sum()
        loss_sil += l_m
        ious = batch_mask_iou(image, self.ref_mask_object)
        l_chamfer = torch.sum(self.compute_edges(image) * self.edt_ref_edge)
        l_chamfer += 100000 * self.compute_offscreen_loss(verts)
        return {
            "loss_sil_obj": loss_sil / len(verts), 
            "loss_edge_obj": l_chamfer,
        }, {
            'iou_object': ious.mean().item()
        }

    def compute_correspondence_loss(self, verts, faces, rotation_obj, translation_obj, correspondence_frame_idxs, correspondence_uvs):
        """
        verts: B * V * 3
        faces: B * F * 3
        rotation_obj: B * 3 * 3
        translation_obj: B * 1 * 3
        correspondence_frame_idxs: N * 2 对应图像的索引
        correspondence_uvs: N * 2 * 5 * 2 对应匹配点的坐标,暂时指定为5个匹配点
        """
        # back transform the camera matrix
        K = self.camintr[0].clone().detach()
        image_size = max(self.full_mask_object.shape[1], self.full_mask_object.shape[2])
        K[:2] = K[:2] * image_size
        K_inv = torch.linalg.inv(K)
        obj_f = faces[0] # F * 3
        rot1, rot2 = rotation_obj[correspondence_frame_idxs[:, 0]], rotation_obj[correspondence_frame_idxs[:, 1]] # N * 3 * 3
        trans1, trans2 = translation_obj[correspondence_frame_idxs[:, 0]], translation_obj[correspondence_frame_idxs[:, 1]] # N * 1 * 3
        pixels_x1, pixels_x2 = correspondence_uvs[:, 0, :, 0], correspondence_uvs[:, 1, :, 0] # N * Point_Num
        pixels_y1, pixels_y2 = correspondence_uvs[:, 0, :, 1], correspondence_uvs[:, 1, :, 1] # N * Point_Num
        rays_o1, rays_d1 = get_rays(pixels_x1.reshape(-1), pixels_y1.reshape(-1), K_inv, torch.eye(4).to(K_inv.device)) # (N * Point_Num) * 3
        rays_o2, rays_d2 = get_rays(pixels_x2.reshape(-1), pixels_y2.reshape(-1), K_inv, torch.eye(4).to(K_inv.device))
        rays_o1, rays_o2 = rays_o1.reshape(-1, pixels_x1.shape[1], 3), rays_o2.reshape(-1, pixels_x1.shape[1], 3) # N * Point_Num * 3
        rays_d1, rays_d2 = rays_d1.reshape(-1, pixels_x1.shape[1], 3), rays_d2.reshape(-1, pixels_x1.shape[1], 3)
        hit_pos1 = []
        hit_pos2 = []
        hit_mask1 = []
        hit_mask2 = []
        # TODO: 处理没有hit的点
        for n_idx in range(rot1.shape[0]): # 0 ~ N-1
            verts1, verts2 = verts[correspondence_frame_idxs[n_idx, 0]], verts[correspondence_frame_idxs[n_idx, 1]]
            matrix1 = torch.cat([rot1[n_idx], trans1[n_idx].reshape(3, 1)], dim=1)
            matrix1 = torch.cat([matrix1, torch.tensor([[0, 0, 0, 1.]]).to(matrix1.device)])
            hit_pos1_, hit_mask1_ = ray_mesh_intersect(verts1, obj_f, rays_o1[n_idx], rays_d1[n_idx], torch.linalg.inv(matrix1))
            matrix2 = torch.cat([rot2[n_idx], trans2[n_idx].reshape(3, 1)], dim=1)
            matrix2 = torch.cat([matrix2, torch.tensor([[0, 0, 0, 1.]]).to(matrix2.device)])
            hit_pos2_, hit_mask2_ = ray_mesh_intersect(verts2, obj_f, rays_o2[n_idx], rays_d2[n_idx], torch.linalg.inv(matrix2))
            # transform back to other camera coordinate
            hit_pos1_2 = hit_pos1_ @ matrix2[:3, :3].T + matrix2[:3, -1].view(1, 3)
            hit_pos2_1 = hit_pos2_ @ matrix1[:3, :3].T + matrix1[:3, -1].view(1, 3)
            hit_pos1.append(hit_pos1_2)
            hit_mask1.append(hit_mask1_)
            hit_pos2.append(hit_pos2_1)
            hit_mask2.append(hit_mask2_)
        hit_pos1 = torch.cat(hit_pos1)
        hit_pos2 = torch.cat(hit_pos2)
        hit_mask1 = torch.cat(hit_mask1)
        hit_mask2 = torch.cat(hit_mask2)
        hit_uv1 = hit_pos1 @ K.T
        hit_uv1 = hit_uv1[:, :2] / hit_uv1[:, 2:]
        hit_uv2 = hit_pos2 @ K.T
        hit_uv2 = hit_uv2[:, :2] / hit_uv2[:, 2:]
        gt_uv2 = torch.cat([pixels_x2.reshape(-1, 1), pixels_y2.reshape(-1, 1)], dim=1).float()
        gt_uv1 = torch.cat([pixels_x1.reshape(-1, 1), pixels_y1.reshape(-1, 1)], dim=1).float()
        # normalize the coordinate
        hit_uv1[:, 0] /= self.full_mask_object.shape[2]; hit_uv1[:, 1] /= self.full_mask_object.shape[1]
        hit_uv2[:, 0] /= self.full_mask_object.shape[2]; hit_uv2[:, 1] /= self.full_mask_object.shape[1]
        gt_uv1[:, 0] /= self.full_mask_object.shape[2]; gt_uv1[:, 1] /= self.full_mask_object.shape[1]
        gt_uv2[:, 0] /= self.full_mask_object.shape[2]; gt_uv2[:, 1] /= self.full_mask_object.shape[1]
        correspondence_loss = torch.sum((hit_uv1 - gt_uv2)**2 * hit_mask1.reshape(-1, 1)) + \
                                torch.sum((hit_uv2 - gt_uv1)**2 * hit_mask2.reshape(-1, 1))
        return {
            "loss_correspondence_obj": correspondence_loss
        }

    def compute_flow_loss(self, verts, faces):
        """
        verts: B * V * 3
        faces: B * F * 3
        """
        face_visibility = self.renderer(verts, faces, mode="visibility")[:, :faces.shape[1]].bool() # B * F
        # 每一帧面片的是否可视
        face_visibility = torch.logical_and(face_visibility[:-1], face_visibility[1:])
        uv = project.batch_proj2d(verts, self.renderer.K)
        # 此处的K由于是在NDC坐标系中，投影坐标范围为[0, 1]
        former_uv = uv[:-1].detach()
        pred_flow = uv[1:] - former_uv # bs * N * 2
        flow_mask = torch.zeros((verts.shape[0] - 1, verts.shape[1])).to(verts.device)
        # TODO: 能否优化不使用for循环
        for i in range(face_visibility.shape[0] - 1):
            obj_vis_v_idx = faces[0, torch.logical_and(face_visibility[i], face_visibility[i+1])].reshape(-1)
            flow_mask[i][torch.unique(obj_vis_v_idx)] = 1
        # 只计算投影在物体mask范围内
        mask_region = F.grid_sample(self.full_mask_object[:-1].unsqueeze(-1).float().permute(0, 3, 1, 2), (2 * former_uv - 1.).clamp(-1, 1).unsqueeze(2), align_corners=False)
        mask_region = mask_region[:, :, :, 0].transpose(1, 2).squeeze(-1)
        flow_mask = torch.logical_and(mask_region==1, flow_mask)
        # sample the gt flows
        sampled_gt_flow = F.grid_sample(self.flow_object.permute(0, 3, 1, 2), (2 * former_uv - 1.).clamp(-1, 1).unsqueeze(2), align_corners=False) # 水平位移和垂直位移
        sampled_gt_flow = sampled_gt_flow[:, :, :, 0].transpose(1, 2) # bs * N * 2
        # 利用方向的cos夹角值来作监督（分别计算x方向和y方向的损失）
        pred_flow_x, sampled_gt_flow_x = pred_flow[:, :, 0], sampled_gt_flow[:, :, 0]
        pred_flow_y, sampled_gt_flow_y = pred_flow[:, :, 1], sampled_gt_flow[:, :, 1]
        cos_x = pred_flow_x * sampled_gt_flow_x / (torch.abs(pred_flow_x) * torch.abs(sampled_gt_flow_x) + 1e-8)
        cos_y = pred_flow_y * sampled_gt_flow_y / (torch.abs(pred_flow_y) * torch.abs(sampled_gt_flow_y) + 1e-8)
        loss_flow_x = torch.sum((1 - cos_x) * flow_mask) / (flow_mask.sum() + 1e-8)
        loss_flow_y = torch.sum((1 - cos_y) * flow_mask) / (flow_mask.sum() + 1e-8)
        loss_flow = loss_flow_x + loss_flow_y
        return {
            "loss_flow_obj": loss_flow
        }


    @staticmethod
    def _compute_iou_1d(mask1, mask2):
        """
        mask1: (2).
        mask2: (2).
        """
        o_l = torch.min(mask1[0], mask2[0])
        o_r = torch.max(mask1[1], mask2[1])
        i_l = torch.max(mask1[0], mask2[0])
        i_r = torch.min(mask1[1], mask2[1])
        inter = torch.clamp(i_r - i_l, min=0)
        return inter / (o_r - o_l)


class Losses():
    def __init__(
        self,
        renderer,
        ref_mask_object,
        ref_verts2d_hand,
        keep_mask_object,
        ref_mask_hand,
        keep_mask_hand,
        camintr_rois_object,
        camintr_rois_hand,
        camintr,
        class_name,
        inter_type="min",
        hand_nb=1,
    ):
        """
        Args:
            inter_type (str): [centroid|min] centroid to penalize centroid distances
                min to penalize minimum distance
        """
        self.renderer = nr.renderer.Renderer(image_size=REND_SIZE,
                                             K=renderer.K,
                                             R=renderer.R,
                                             t=renderer.t,
                                             orig_size=1)
        self.inter_type = inter_type
        self.ref_mask_object = ref_mask_object
        self.keep_mask_object = keep_mask_object
        self.camintr_rois_object = camintr_rois_object
        self.ref_mask_hand = ref_mask_hand
        self.ref_verts2d_hand = ref_verts2d_hand
        self.keep_mask_hand = keep_mask_hand
        self.camintr_rois_hand = camintr_rois_hand
        # Necessary ! Otherwise camintr gets updated for some reason TODO check
        self.camintr = camintr.clone()
        self.thresh = 3  # z thresh for interaction loss
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name
        self.hand_nb = hand_nb

        self.expansion = 0.2
        self.interaction_map = INTERACTION_MAPPING[class_name]

        self.interaction_pairs = None

    def assign_interaction_pairs(self, verts_hand, verts_object):
        """
        Assigns pairs of people and objects that are interacting.
        Unlike PHOSA, one obect can be assigned to multiple hands but one hand is
        only assigned to a given object (e.g. we assume that a hand is
        not manipulating two obects at the same time)

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A person and an object are interacting if the 3D bounding boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_hand (B x V_p x 3).
            verts_object (B x V_o x 3).

        Returns:
            interaction_pairs: List[Tuple(person_index, object_index)]
        """
        interacting = []

        with torch.no_grad():
            bboxes_object = project_bbox(verts_object,
                                         self.renderer,
                                         bbox_expansion=self.expansion)
            bboxes_hand = project_bbox(verts_hand,
                                       self.renderer,
                                       bbox_expansion=self.expansion)
            for batch_idx, (box_obj, box_hand) in enumerate(
                    zip(bboxes_object, bboxes_hand)):
                iou = compute_iou(box_obj, box_hand)
                z_dist = compute_dist_z(verts_object[batch_idx],
                                        verts_hand[batch_idx])
                if (iou > 0) and (z_dist < self.thresh):
                    interacting.append(1)
                else:
                    interacting.append(0)
            return interacting

    def compute_verts2d_loss_hand(self,
                                  verts,
                                  image_size=640,
                                  min_hand_size=70):
        camintr = self.camintr.unsqueeze(1).repeat(1, self.hand_nb, 1,
                                                   1).view(-1, 3, 3)
        pred_verts_proj = project.batch_proj2d(verts, camintr)
        tar_verts = self.ref_verts2d_hand / image_size
        verts2d_loss = ((pred_verts_proj - tar_verts)**2).sum(-1).mean()
        too_small_hands = (self.ref_verts2d_hand -
                           self.ref_verts2d_hand.mean(1).unsqueeze(1)).norm(
                               2, -1).max(1)[0] < min_hand_size
        verts2d_loss_new = (
            ((pred_verts_proj - tar_verts)**2) *
            (1 - too_small_hands.float()).unsqueeze(1).unsqueeze(1)
        ).sum(-1).mean()
        verts2d_dist = (pred_verts_proj * image_size -
                        self.ref_verts2d_hand).norm(2, -1).mean()
        # HACK TODO beautify, discard hands that are too small !
        return {
            "loss_v2d_hand": verts2d_loss
        }, {
            "v2d_hand": verts2d_dist.item()
        }

    def compute_sil_loss_hand(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        for i in range(len(verts)):
            verts = verts[i].unsqueeze(0)
            camintr = self.camintr_rois_hand[i]
            # Rendering happens in ROI
            rend = self.renderer(
                verts,  # TODO check why not verts[i]
                faces[i],
                K=camintr.unsqueeze(0),
                mode="silhouettes")
            image = self.keep_mask_hand[i] * rend
            l_m = torch.sum((image - self.ref_mask_hand[i])**
                            2) / self.keep_mask_hand[i].sum()
            loss_sil += l_m
        return {"loss_sil_hand": loss_sil / len(verts)}

    def compute_sil_loss_object(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        # Rendering happens in ROI
        camintr = self.camintr_rois_object
        rend = self.renderer(verts, faces, K=camintr, mode="silhouettes")
        image = self.keep_mask_object * rend
        l_m = torch.sum(
            (image - self.ref_mask_object)**2) / self.keep_mask_object.sum()
        loss_sil += l_m
        ious = batch_mask_iou(image, self.ref_mask_object)
        return {
            "loss_sil_obj": loss_sil / len(verts)
        }, {
            'iou_object': ious.mean().item()
        }

    def compute_interaction_loss(self, verts_hand_b, verts_object_b):
        """
        Computes interaction loss.
        Args:
            verts_hand_b (B, person_nb, vert_nb, 3)
            verts_object_b (B, object_nb, vert_nb, 3)
        """
        loss_inter = torch.Tensor([0.0]).float().cuda()
        num_interactions = 0
        min_dists = []
        for person_idx in range(verts_hand_b.shape[1]):
            for object_idx in range(verts_object_b.shape[1]):
                interacting = self.assign_interaction_pairs(
                    verts_hand_b[:, person_idx], verts_object_b[:, object_idx])
                for batch_idx, inter in enumerate(interacting):
                    if inter:
                        v_p = verts_hand_b[batch_idx, person_idx]
                        v_o = verts_object_b[batch_idx, object_idx]
                        if self.inter_type == "centroid":
                            inter_error = self.mse(v_p.mean(0), v_o.mean(0))
                        elif self.inter_type == "min":
                            inter_error = distutils.batch_pairwise_dist(
                                v_p.unsqueeze(0), v_o.unsqueeze(0)).min()
                        loss_inter += inter_error
                        num_interactions += 1
                # Compute minimal vertex distance
                with torch.no_grad():
                    min_dist = torch.sqrt(
                        distutils.batch_pairwise_dist(
                            verts_hand_b[:, person_idx],
                            verts_object_b[:, object_idx])).min(1)[0].min(1)[0]
                min_dists.append(min_dist)

        # Avoid nans by 0 division
        if num_interactions > 0:
            loss_inter_ = loss_inter / num_interactions
        else:
            loss_inter = loss_inter
        min_dists = torch.stack(min_dists).min(0)[0]
        return {
            "loss_inter": loss_inter
        }, {
            "handobj_maxdist": torch.max(min_dists).item()
        }

    @staticmethod
    def _compute_iou_1d(mask1, mask2):
        """
        mask1: (2).
        mask2: (2).
        """
        o_l = torch.min(mask1[0], mask2[0])
        o_r = torch.max(mask1[1], mask2[1])
        i_l = torch.max(mask1[0], mask2[0])
        i_r = torch.min(mask1[1], mask2[1])
        inter = torch.clamp(i_r - i_l, min=0)
        return inter / (o_r - o_l)
