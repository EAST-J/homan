# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member,wrong-import-order,too-many-branches,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from homan import lossutils
from homan.losses import Losses, Losses_Obj
from homan.lossutils import compute_ordinal_depth_loss
from homan.manomodel import ManoModel
from homan.meshutils import get_faces_and_textures
from homan.utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
)
from homan.utils.geometry import combine_verts, matrix_to_rot6d, rot6d_to_matrix

from libyana.conversions import npt
from libyana.lib3d import trans3d
from libyana.verify.checkshape import check_shape


class HOMan_Obj(nn.Module):
    def __init__(
        self,
        translations_object,
        rotations_object,
        verts_object_og,
        faces_object,
        masks_object,
        camintr_rois_object,
        target_masks_object,
        flow_object,
        class_name,
        int_scale_init=1.0,
        camintr=None,
        optimize_object_scale=False,
        inter_type="centroid",
        image_size=640,
    ):
        super().__init__()
        # Initialize object pamaters
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init,
                                                requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        if rotations_object.shape[-1] == 3:
            rotations_object6d = matrix_to_rot6d(rotations_object)
        else:
            rotations_object6d = rotations_object
        self.obj_rot_mult = 1  # This scaling has no effect !
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.register_buffer("verts_object_og", verts_object_og)
        # Inititalize person parameters

        init_scales = int_scale_init * torch.ones(1).float()
        init_scales_mean = torch.Tensor(int_scale_init).float()
        self.optimize_object_scale = optimize_object_scale
        if optimize_object_scale:
            self.int_scales_object = nn.Parameter(
                init_scales,
                requires_grad=True,
            )
        else:
            self.register_buffer("int_scales_object", init_scales)
        self.register_buffer("int_scale_object_mean", torch.ones(1).float())
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.register_buffer("flow_object",
                             flow_object.float())
        self.register_buffer("camintr_rois_object", camintr_rois_object)
        self.register_buffer("faces_object", faces_object)
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], faces_object.shape[1], 1, 1, 1,
                       3))
        self.cuda()

        # Setup renderer
        if camintr is None:
            camintr = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5],
                                               [0, 0, 1]]])
        else:
            camintr = npt.tensorify(camintr)
            if camintr.dim() == 2:
                camintr = camintr.unsqueeze(0)
            camintr = camintr.cuda().float()
        rot = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        trans = torch.zeros(1, 3).cuda()
        self.register_buffer("camintr", camintr)
        self.image_size = image_size
        self.renderer = nr.renderer.Renderer(image_size=self.image_size,
                                             K=camintr.clone(),
                                             R=rot,
                                             t=trans,
                                             orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]
        if masks_object.dim() == 2:
            masks_object = masks_object.unsqueeze(0)
        self.register_buffer("masks_object", masks_object)
        verts_object, _ = self.get_verts_object()
        ref_verts_list = [verts_object[:1]]
        ref_faces_list = [faces_object[:1]]
        pred_colors = ["gold"] 
        gt_colors = ["green"] 
        faces, textures = get_faces_and_textures(ref_verts_list,
                                                 ref_faces_list,
                                                 color_names=pred_colors)
        # Assumes only one object
        batch_size = verts_object.shape[0]
        self.faces = faces.repeat(batch_size, 1, 1)
        self.textures = textures.repeat(batch_size, 1, 1, 1, 1, 1)
        faces_gt, textures_gt = get_faces_and_textures(ref_verts_list,
                                                       ref_faces_list,
                                                       color_names=gt_colors)
        self.textures_gt = textures_gt.repeat(batch_size, 1, 1, 1, 1, 1)
        self.faces_gt = faces_gt.repeat(batch_size, 1, 1)

        faces_with_gt, textures_with_gt = get_faces_and_textures(
            ref_verts_list + ref_verts_list,
            ref_faces_list + ref_faces_list,
            color_names=pred_colors + gt_colors)

        self.textures_with_gt = textures_with_gt.repeat(
            batch_size, 1, 1, 1, 1, 1)
        self.faces_with_gt = faces_with_gt.repeat(batch_size, 1, 1)
        self.losses = Losses_Obj(
            renderer=self.renderer,
            ref_mask_object=self.ref_mask_object,
            keep_mask_object=self.keep_mask_object,
            full_mask_object=self.masks_object,
            flow_object=self.flow_object,
            camintr_rois_object=self.camintr_rois_object,
            camintr=self.camintr,
            class_name=class_name,
            inter_type=inter_type,
        )
        verts_object_init, _ = self.get_verts_object()
        self.verts_object_init = verts_object_init.detach().clone()


    def get_verts_object(self):
        rotations_object = rot6d_to_matrix(self.obj_rot_mult *
                                           self.rotations_object)
        obj_verts = compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=self.translations_object,
            rotations=rotations_object,
            intrinsic_scales=self.int_scales_object.abs(),
        )
        return obj_verts


    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss_dict = {}
        metric_dict = {}
        verts_object, _ = self.get_verts_object()
        if loss_weights is None or ((loss_weights["lw_smooth_hand"] > 0) or
                                    (loss_weights["lw_smooth_obj"] > 0)):
            loss_smooth = lossutils.compute_smooth_loss(
                verts_hand=None,
                verts_obj=verts_object,
            )
            loss_dict.update(loss_smooth)
        if loss_weights is None or loss_weights["lw_collision"] > 0:
            # Pushes hand out of object, gradient not flowing through object !
            loss_coll = 0
            loss_dict.update(loss_coll)

        if loss_weights is None or loss_weights["lw_contact"] > 0:
            loss_contact, _ = 0
            loss_dict.update(loss_contact)
        # if loss_weights is None or loss_weights["lw_v2d_hand"] > 0:
        #     if self.optimize_object_scale:
        #         loss_verts2d, metric_verts2d = self.losses.compute_verts2d_loss_hand(
        #             verts=verts_hand,
        #             image_size=self.image_size,
        #             min_hand_size=70)
        #     else:
        #         loss_verts2d, metric_verts2d = self.losses.compute_verts2d_loss_hand(
        #             verts=verts_hand,
        #             image_size=self.image_size,
        #             min_hand_size=1000)
        #     loss_dict.update(loss_verts2d)
        #     metric_dict.update(metric_verts2d)
        if loss_weights is None or loss_weights["lw_flow_obj"] > 0:
            flow_loss_dict = self.losses.compute_flow_loss(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(flow_loss_dict)
        if loss_weights is None or loss_weights["lw_sil_obj"] > 0:
            sil_loss_dict, sil_metric_dict = self.losses.compute_sil_loss_object(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(sil_loss_dict)
            metric_dict.update(sil_metric_dict)

            # loss_dict.update(
            #     self.losses.compute_sil_loss_hand(verts=verts_hand,
            #                                         faces=[self.faces_hand] *
            #                                         len(verts_hand)))
        # if loss_weights is None or loss_weights["lw_inter"] > 0:
        #     # Interaction acts only on hand !
        #     if not self.optimize_object_scale:
        #         inter_verts_object = verts_object.unsqueeze(1).detach()
        #     else:
        #         inter_verts_object = verts_object.unsqueeze(1)
        #     inter_loss_dict, inter_metric_dict = self.losses.compute_interaction_loss(
        #         verts_hand_b=verts_hand_det.view(-1, self.hand_nb, 778, 3),
        #         verts_object_b=inter_verts_object)
        #     loss_dict.update(inter_loss_dict)
        #     metric_dict.update(inter_metric_dict)

        if loss_weights is None or loss_weights["lw_scale_obj"] > 0:
            loss_dict[
                "loss_scale_obj"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.int_scales_object,
                    intrinsic_mean=self.int_scale_object_mean,
                )
        # if loss_weights is None or loss_weights["lw_scale_hand"] > 0:
        #     loss_dict[
        #         "loss_scale_hand"] = lossutils.compute_intrinsic_scale_prior(
        #             intrinsic_scales=self.int_scales_hand,
        #             intrinsic_mean=self.int_scale_hand_mean,
        #         )
        if loss_weights is None or loss_weights["lw_depth"] > 0:
            loss_dict.update(lossutils.compute_ordinal_depth_loss())
        return loss_dict, metric_dict

    def render_limem(self,
                     renderer,
                     verts,
                     faces,
                     textures,
                     K,
                     max_in_batch=5):
        sample_nb = verts.shape[0]
        check_shape(verts, (-1, -1, 3))
        check_shape(faces, (sample_nb, -1, 3))
        check_shape(textures, (sample_nb, -1, 1, 1, 1, 3))
        check_shape(K, (sample_nb, 3, 3))

        if max_in_batch is not None:
            chunk_nb = (sample_nb + 1) // min(max_in_batch, sample_nb)
        else:
            chunk_nb = 1
        verts_chunks = verts.chunk(chunk_nb, 0)
        faces_chunks = faces.chunk(chunk_nb, 0)
        textures_chunks = textures.chunk(chunk_nb, 0)
        K_chunks = K.chunk(chunk_nb, 0)
        all_images = []
        all_masks = []
        for vert, face, tex, camintr in zip(verts_chunks, faces_chunks,
                                            textures_chunks, K_chunks):
            chunk_images, _, chunk_masks = renderer.render(vertices=vert,
                                                           faces=face,
                                                           textures=tex,
                                                           K=camintr)
            all_images.append(
                np.clip(npt.numpify(chunk_images).transpose(0, 2, 3, 1), 0, 1))
            all_masks.append(npt.numpify(chunk_masks).astype(bool))
        all_images = np.concatenate(all_images)
        all_masks = np.concatenate(all_masks)
        return all_images, all_masks

    def render(self, renderer, rotate=False, viz_len=10, max_in_batch=None):
        verts_object = self.get_verts_object()[0]
        verts_combined = combine_verts([verts_object])
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces[:viz_len],
                                          self.textures[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def render_gt(self,
                  renderer,
                  verts_object_gt=None,
                  rotate=False,
                  viz_len=10,
                  max_in_batch=None):
        verts_list = [verts_object_gt]
        verts_combined = combine_verts(verts_list)
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces[:viz_len],
                                          self.textures_gt[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def render_with_gt(self,
                       renderer,
                       verts_hand_gt=None,
                       verts_object_gt=None,
                       rotate=False,
                       viz_len=10,
                       init=False,
                       max_in_batch=None):
        if init:
            verts_object_pred = self.verts_object_init
        else:
            verts_object_pred = self.get_verts_object()[0]
        verts_list = [verts_object_pred
                      ] + [verts_object_gt]
        verts_combined = combine_verts(verts_list)
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces_with_gt[:viz_len],
                                          self.textures_with_gt[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def save_obj(self, fname):
        with open(fname, "w") as fp:
            verts_combined = combine_verts(
                [self.get_verts_object()[0],
                 self.get_verts_hand()[0]])
            for v in tqdm.tqdm(verts_combined[0]):
                fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
            o = 1
            for face in tqdm.tqdm(self.faces[0]):
                fp.write(
                    f"f {face[0] + o:d} {face[1] + o:d} {face[2] + o:d}\n")
