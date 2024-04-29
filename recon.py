# The homan demo code: reconstruct the hand and object with input images and object CAD model
import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, "detectors/hand_object_detector/lib")
sys.path.insert(0, "external/frankmocap")


import numpy as np
from PIL import Image
import trimesh 

from handmocap.hand_bbox_detector import HandBboxDetector

from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh
# Load images
image_folder = "images"
image_names = sorted(os.listdir(image_folder))

# start_idx = 160
# if start_idx >= len(image_names) - 10:
#   start_idx = len(image_names) - 11
# # 10 images for sample
# image_paths = [os.path.join(image_folder, image_name) for image_name in image_names[start_idx:start_idx + 10]]
image_paths = [os.path.join(image_folder, image_name) for image_name in image_names if (image_name.lower().endswith(".jpg") or image_name.lower().endswith(".png"))][::5]
print(image_paths)
print(len(image_paths))

exp_name = "pred"
if exp_name == "gt":
    obj_path = "local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj"
else:
    obj_path = "/remote-home/jiangshijian/shap-e/example_mesh_0.ply"
# Initialize object scale
obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm)
obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
obj_faces = np.array(obj_mesh.faces)
# Convert images to numpy 
images = [Image.open(image_path) for image_path in image_paths]
images_np = [np.array(image) for image in images]

# Load object mesh
hand_detector = get_hand_bbox_detector()
seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})
hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
obj_bboxes = [seq_boxes['objects']]


# get 2D masks and 3D hand pose
from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from handmocap.hand_mocap_api import HandMocap


sample_folder = os.path.join("tmp/", exp_name)
os.makedirs(sample_folder, exist_ok=True)

# Initialize segmentation and hand pose estimation models
mask_extractor = MaskExtractor(pointrend_model_weights="/remote-home/jiangshijian/homan/external/model_final_edd263.pkl") # detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl
frankmocap_hand_checkpoint = "/remote-home/jiangshijian/homan/external/frankmocap/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
hand_predictor = HandMocap(frankmocap_hand_checkpoint, "extra_data/smpl")

# Define camera parameters
height, width, _ = images_np[0].shape
image_size = max(height, width)
focal = 480
camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
camintrs = [camintr for _ in range(len(images_np))]


person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(images_np,
                                                                  mask_extractor=mask_extractor,
                                                                  hand_predictor=hand_predictor,
                                                                  hand_bboxes=hand_bboxes,
                                                                  obj_bboxes=np.stack(obj_bboxes),
                                                                  sample_folder=sample_folder,
                                                                  camintr=camintrs,
                                                                  image_size=image_size,
                                                                  debug=False)

tmp_ = Image.fromarray(super2d_imgs)
tmp_.save(os.path.join(sample_folder, 'tmp_2d.png'))


from homan.pose_optimization import find_optimal_poses
from homan.lib2d import maskutils

# ! Step1: Initialize object motion, based on the object masks
object_parameters = find_optimal_poses(
    images=images_np,
    image_size=images_np[0].shape,
    vertices=obj_verts_can,
    faces=obj_faces,
    annotations=obj_mask_infos,
    num_initializations=400,
    num_iterations=10, # Increase to get more accurate initializations
    Ks=np.stack(camintrs),
    viz_path=os.path.join(sample_folder, "optimal_pose.png"),
    debug=False,
)
# print(person_parameters)
# print(object_parameters)
# Add object object occlusions to hand masks
for person_param, obj_param, camintr in zip(person_parameters,
                                        object_parameters,
                                        camintrs):
    
    maskutils.add_target_hand_occlusions(
        person_param,
        obj_param,
        camintr,
        debug=False,
        sample_folder=sample_folder)
    
# ! Step2: joint optimize the object pose and hand pose, using motion smoothness and hand-object proximity as prior
from homan.viz.colabutils import display_video
from homan.jointopt import optimize_hand_object

coarse_num_iterations = 201 # Increase to give more steps to converge
coarse_viz_step = 10 # Decrease to visualize more optimization steps
coarse_loss_weights = {
        "lw_inter": 1,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.0,
        "lw_contact": 0.0,
        "lw_scale_hand": 0.001,
        "lw_scale_obj": 0.001,
        "lw_v2d_hand": 50,
        "lw_smooth_hand": 2000,
        "lw_smooth_obj": 2000,
        "lw_pca": 0.004,
    }

# Camera intrinsics in normalized coordinate
camintr_nc = np.stack(camintrs).copy().astype(np.float32)
camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

step2_folder = os.path.join(sample_folder, "jointoptim_step2")
step2_viz_folder = os.path.join(step2_folder, "viz")

# Coarse hand-object fitting
model, loss_evolution, imgs = optimize_hand_object(
    person_parameters=person_parameters,
    object_parameters=object_parameters,
    hand_proj_mode="persp",
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_mano=True,
    optimize_object_scale=True,
    loss_weights=coarse_loss_weights,
    image_size=image_size,
    num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=None,
    viz_step=coarse_viz_step,
    viz_folder=step2_viz_folder,
)

last_viz_idx = (coarse_num_iterations // coarse_viz_step) * coarse_viz_step
# video_step2 = display_video(os.path.join(step2_folder, "joint_optim.mp4"),
#                             os.path.join(sample_folder, "jointoptim_step2.mp4"))

#########################################
# ! tmp code to get the hand & object parameters
'''
Parameter Lists:
mano_pca_pose: B * 45
mano_rot: 估计的mano全局旋转参数 B * 3
mano_trans: 估计的mano全局平移参数 B * 3
mano_betas: B * 10
rotation_hand: 估计的在mano坐标系和相机坐标系之间的旋转矩阵 B * 3 * 2
translation_hand: 估计的在mano坐标系和相机坐标系之间的旋转矩阵 B * 1 * 3

verts_object_og: N * 3
int_scales_object: tensor([])
rotations_object: B * 3 * 2
translations_object: B * 1 * 3
faces_object: B * F * 3

for object, R: rot6d_to_matrix(model.rotations_object).T T:model.translations_object
K: focal, 0, width//2
    0, focal, height//2
    0, 0, 1
'''
# get the predict hand_verts
# import trimesh
# from homan.utils.geometry import rot6d_to_matrix
# from homan.utils.camera import compute_transformation_persp
# from renderer import Visualizer
# import cv2

# vis = Visualizer((height, width))
# obj_v, _ = model.get_verts_object()
# obj_v = obj_v[0].detach().cpu()

# res_img = vis.draw_mesh(images_np[0], obj_v, model.faces_object[0].detach().cpu().numpy(), (focal, focal, width // 2, height // 2))
# cv2.imwrite('tmp.jpg', res_img)
# # trimesh.Trimesh(obj_v, model.faces_object[0].detach().cpu().numpy()).export('tmp_obj.ply')

# obj_v_og = model.verts_object_og
# rot_m = rot6d_to_matrix(model.rotations_object[0:1])[0]
# obj_scale = model.int_scales_object
# trans_obj = model.translations_object[0]
# ! rot_m在rot6d_to_matrix中与一般的方式有些略微的区别，即这里获得的是旋转矩阵的转置
# obj_v_ = (obj_scale * obj_v_og) @ rot_m + trans_obj
# print(obj_v.detach().cpu() - obj_v)
# hand_v, _ = model.get_verts_hand()
# hand_v = hand_v[0].detach().cpu().numpy()

# import cv2
# import torch
# import copy
# from manopth.manolayer import ManoLayer
# hand_model = ManoLayer(mano_root='/remote-home/jiangshijian/homan/extra_data/mano', use_pca=True, flat_hand_mean=False, ncomps=16)
# trimesh.Trimesh(hand_v, hand_model.th_faces.detach().cpu().numpy()).export('tmp_hand_.ply')


# pose = model.mano_pca_pose[0:1]
# betas = model.mano_betas[0:1]
# rot = model.mano_rot[0:1]
# trans = model.mano_trans[0:1]
# rotation_hand = rot6d_to_matrix(model.rotations_hand[0:1])
# translation_hand = model.translations_hand[0:1]
# theta = torch.cat([rot, pose], dim=1)
# hand_model = hand_model.to(pose.device)
# pred_v, pred_j = hand_model(theta, betas, trans)
# pred_v = pred_v[0] / 1000.
# pred_j = pred_j[0] / 1000.
# pred_v, _ = compute_transformation_persp(pred_v, translation_hand, rotation_hand)
# pred_v = pred_v.detach().cpu().numpy()
# pred_j = pred_j.detach().cpu().numpy()
# trimesh.Trimesh(pred_v, hand_model.th_faces.detach().cpu().numpy()).export('tmp_hand.ply')
# print(hand_v - pred_v)
#########################################

# ! Step3: refine the poses
finegrained_num_iterations = 401   # Increase to give more time for convergence
finegrained_loss_weights = {
        "lw_inter": 1,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.001,
        "lw_contact": 1.0,
        "lw_scale_hand": 0.001,
        "lw_scale_obj": 0.001,
        "lw_v2d_hand": 50,
        "lw_smooth_hand": 2000,
        "lw_smooth_obj": 2000,
        "lw_pca": 0.004,
    }
finegrained_viz_step = 10 # Decrease to visualize more optimization steps

# Refine hand-object fitting
step3_folder = os.path.join(sample_folder, "jointoptim_step3")
step3_viz_folder = os.path.join(step3_folder, "viz")
model_fine, loss_evolution, imgs = optimize_hand_object(
    person_parameters=person_parameters,
    object_parameters=object_parameters,
    hand_proj_mode="persp",
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_mano=True,
    optimize_object_scale=True,
    loss_weights=finegrained_loss_weights,
    image_size=image_size,
    num_iterations=finegrained_num_iterations + 1,
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=model.state_dict(),
    viz_step=finegrained_viz_step, 
    viz_folder=step3_viz_folder,
)
last_viz_idx = (finegrained_num_iterations // finegrained_viz_step) * finegrained_viz_step
# video_step3 = display_video(os.path.join(step3_folder, "joint_optim.mp4"),
#                             os.path.join(sample_folder, "jointoptim_step3.mp4"))

    



