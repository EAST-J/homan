# The homan demo code: reconstruct the hand and object with input images and object CAD model
import os
import sys
import torch
import point_cloud_utils as pcu
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, "detectors/hand_object_detector/lib")
sys.path.insert(0, "external/frankmocap")


import numpy as np
from PIL import Image
import trimesh 

from handmocap.hand_bbox_detector import HandBboxDetector
from detectron2.structures import BitMasks
from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh, bbox_wh_to_xy
from homan.pose_optimization import find_optimal_poses
from homan.lib2d import maskutils


def process_masks_to_infos(obj_masks, hand_masks):
    # 使用预先获得的mask，处理为detectron2的数据格式
    obj_mask_infos = []
    for obj_mask, hand_mask in zip(obj_masks, hand_masks):
        obj_mask = (obj_mask == 255)
        hand_mask = (hand_mask == 255)
        hand_occlusions = torch.from_numpy(hand_mask).unsqueeze(0)
        image_size = obj_mask.shape
        bit_masks = BitMasks(torch.from_numpy(obj_mask).unsqueeze(0))
        full_boxes = torch.tensor([[0, 0, image_size[1], image_size[0]]] *
                                  1).float()
        full_sized_masks = bit_masks.crop_and_resize(full_boxes, image_size[0])
        obj_mask_info = {}
        non_zero_indices = np.nonzero(obj_mask)
        # 获取最小外接矩形的左上角和右下角坐标
        min_row = max(np.min(non_zero_indices[0]) - 5., 0)
        max_row = min(np.max(non_zero_indices[0]) + 5., obj_mask.shape[0])
        min_col = max(np.min(non_zero_indices[1]) -5., 0)
        max_col = min(np.max(non_zero_indices[1]) + 5., obj_mask.shape[1])
        box = torch.tensor([min_col, min_row, max_col, max_row]).float()
        bbox = bbox_xy_to_wh(box)  # xy_wh
        square_bbox = make_bbox_square(bbox, 0.3)
        square_boxes = torch.FloatTensor(
                np.tile(bbox_wh_to_xy(square_bbox),
                        (1, 1)))
        crop_masks = bit_masks.crop_and_resize(square_boxes,
                                                   256).clone().detach()
        obj_mask_info.update({
                "bbox":
                bbox,
                "class_id":
                -1,
                "full_mask":
                full_sized_masks[0, :obj_mask.shape[0], :obj_mask.shape[1]].cpu(),
                "score":
                None,
                "square_bbox":
                square_bbox,  # xy_wh
                "crop_mask":
                crop_masks[0].cpu().numpy(),
        })
        target_masks = maskutils.add_occlusions([obj_mask_info["crop_mask"]],
                                            hand_occlusions,
                                            [obj_mask_info["square_bbox"]])[0]
        obj_mask_info.update({"target_crop_mask": target_masks})
        obj_mask_infos.append(obj_mask_info)
    
    return obj_mask_infos


# Load images
image_folder = "/remote-home/jiangshijian/data/HOI4D/Kettle_1/image"
image_names = sorted(os.listdir(image_folder))

start_idx = 24
if start_idx >= len(image_names) - 10:
  start_idx = len(image_names) - 11
# 10 images for sample
image_paths = [os.path.join(image_folder, image_name) for image_name in image_names[start_idx:start_idx + 10]]
obj_masks_paths = [image_path.replace('image', 'obj_mask') for image_path in image_paths]
hand_masks_paths = [image_path.replace('image', 'hand_mask') for image_path in image_paths]
print(image_paths)
print(obj_masks_paths)
print(hand_masks_paths)

obj_path = "/remote-home/jiangshijian/data/HOI4D/Kettle_1/oObj.obj" # gt_path:/remote-home/jiangshijian/data/HOI4D/Kettle_1/oObj.obj
# Initialize object scale
obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm)
obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
obj_faces = np.array(obj_mesh.faces)
# ! Decimating the mesh in case CUDA out of memory.
obj_verts_can, obj_faces, _, _ = pcu.decimate_triangle_mesh(obj_verts_can, obj_faces, int(obj_faces.shape[0] / 10.))
obj_verts_can = obj_verts_can.astype(np.float32)
# Convert images to numpy 
images = [Image.open(image_path) for image_path in image_paths if (image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"))]
obj_masks = [Image.open(mask_path) for mask_path in obj_masks_paths if (mask_path.lower().endswith(".jpg") or mask_path.lower().endswith(".png"))]
hand_masks = [Image.open(mask_path) for mask_path in hand_masks_paths if (mask_path.lower().endswith(".jpg") or mask_path.lower().endswith(".png"))]
# for idx, img in enumerate(images):
#   images[idx] = img.resize((256, 256))
images_np = [np.array(image) for image in images]
obj_masks_np = [np.array(mask) for mask in obj_masks]
hand_masks_np = [np.array(mask) for mask in hand_masks]
# Load object mesh
hand_detector = get_hand_bbox_detector()
seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})
hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
obj_bboxes = [seq_boxes['objects']]


# get 2D masks and 3D hand pose
from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from handmocap.hand_mocap_api import HandMocap
from homan.viz.vizframeinfo import viz_frame_info


sample_folder = "tmp_hoi4d/"

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

# ! Step1: Initialize object motion, based on the object masks
# obj_mask_infos type: List
obj_mask_infos = process_masks_to_infos(obj_masks_np, hand_masks_np)
person_parameters, _, super2d_imgs = get_frame_infos(images_np,
                                                    mask_extractor=mask_extractor,
                                                    hand_predictor=hand_predictor,
                                                    hand_bboxes=hand_bboxes,
                                                    obj_bboxes=np.stack(obj_bboxes),
                                                    sample_folder=sample_folder,
                                                    camintr=camintrs,
                                                    image_size=image_size,
                                                    debug=False)
for idx, (person_parameter, hand_mask) in enumerate(zip(person_parameters, hand_masks_np)):
    hand_mask = (hand_mask == 255)
    device = person_parameter['masks'].device
    hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).to(device)
    person_parameters[idx]['masks'] = hand_mask

frame_info = {"person_parameters": person_parameters[0], "obj_mask_infos": obj_mask_infos[0], "image": images_np[0]}
super2d_img = viz_frame_info(frame_info,
                                sample_folder=sample_folder,
                                save=False)

tmp_ = Image.fromarray(super2d_img)
tmp_.save('tmp_hoi4d/tmp_2d.png')
# TODO: use the HOI4D provided mask to replace the predicted results





object_parameters = find_optimal_poses(
    images=images_np,
    image_size=images_np[0].shape,
    vertices=obj_verts_can,
    faces=obj_faces,
    annotations=obj_mask_infos,
    num_initializations=200,
    num_iterations=10, # Increase to get more accurate initializations
    Ks=np.stack(camintrs),
    viz_path=os.path.join(sample_folder, "optimal_pose.png"),
    debug=False,
)
print(person_parameters)
print(object_parameters)
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

# ! Step3: refine the poses
finegrained_num_iterations = 201   # Increase to give more time for convergence
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


# TODO: using the model_fine results to get the mesh.