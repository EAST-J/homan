import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, "detectors/hand_object_detector/lib")
sys.path.insert(0, "external/frankmocap")

# To get the object pose parameters
import torch
import numpy as np
from PIL import Image
import trimesh 
import pickle as pkl
from handmocap.hand_bbox_detector import HandBboxDetector
from detectron2.structures import BitMasks
from glob import glob
from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh, bbox_wh_to_xy

from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from handmocap.hand_mocap_api import HandMocap
from homan.viz.vizframeinfo import viz_frame_info
from homan.lib2d import maskutils
import cv2
from torchvision import transforms

# TODO: 如何解决CUDA Memory的问题，考虑对于mask使用crop的mask进行优化？

def process_masks_to_infos(obj_masks, hand_masks):
    # 使用预先获得的mask，处理为detectron2的数据格式
    obj_mask_infos = []
    for obj_mask, hand_mask in zip(obj_masks, hand_masks):
        obj_mask = (obj_mask == 255)
        hand_mask = (hand_mask == 255)
        hand_occlusions = torch.from_numpy(hand_mask).unsqueeze(0)
        image_size = obj_mask.shape
        bit_masks = BitMasks(torch.from_numpy(obj_mask).unsqueeze(0))
        # full_boxes = torch.tensor([[0, 0, image_size[1], image_size[0]]] *
        #                           1).float()
        # full_sized_masks = bit_masks.crop_and_resize(full_boxes, max(image_size[0], image_size[1]))
        full_sized_masks = bit_masks.tensor
        obj_mask_info = {}
        non_zero_indices = np.nonzero(obj_mask)
        # 获取最小外接矩形的左上角和右下角坐标
        min_row = max(np.min(non_zero_indices[0]) - 5., 0)
        max_row = min(np.max(non_zero_indices[0]) + 5., obj_mask.shape[0])
        min_col = max(np.min(non_zero_indices[1]) - 5., 0)
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
                bbox, # bbox为原先图像空间中的包围框，格式为xywh
                "class_id":
                -1,
                "full_mask":
                full_sized_masks[0, :obj_mask.shape[0], :obj_mask.shape[1]].cpu(),
                "score":
                None,
                "square_bbox":
                square_bbox,  # xy_wh 与bbox的中心相同，将包围框给放大
                "crop_mask":
                crop_masks[0].cpu().numpy(),
        })
        # 1 for object part, 0 for background, -1 for hand part.
        target_masks = maskutils.add_occlusions([obj_mask_info["crop_mask"]],
                                            hand_occlusions,
                                            [obj_mask_info["square_bbox"]])[0]
        obj_mask_info.update({"target_crop_mask": target_masks})
        obj_mask_infos.append(obj_mask_info)
    
    return obj_mask_infos

# Load images
seq_name = "SB14"
exp_name ="gt"
data_root = "/remote-home/jiangshijian/data/HO3D_v3/train"
optim_obj_scale = True
start_idx = 900
end_idx = 1100
image_paths = sorted(glob(os.path.join(data_root, seq_name, "rgb", "*.jpg")))[start_idx:end_idx]
# pre_transform = transforms.Compose([])
print(image_paths)
print(len(image_paths))
with open(os.path.join(data_root, seq_name, "meta", "0000.pkl"), "rb") as f:
    data = pkl.load(f, encoding='latin1')
objName = data["objName"]
if exp_name == 'gt':
    obj_path = os.path.join("local_data/datasets/ycbmodels/", objName, "textured_simple_2000.obj")
    obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm)
else:
    # obj_path = "/remote-home/jiangshijian/shap-e/drill.ply"
    obj_path = "/remote-home/jiangshijian/shap-e/cleaner.ply"
    obj_scale = 0.1
# Initialize object scale
obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
obj_faces = np.array(obj_mesh.faces)

# image_transform = transforms.Compose([transforms.Resize((240, 320))])
# mask_transform = transforms.Compose([])
# Convert images to numpy 
images = [(Image.open(image_path)) for image_path in image_paths]
images_np = [np.array(image) for image in images]
masks = [(Image.open(image_path.replace("rgb", "seg")[:-4] + ".png")) for image_path in image_paths] # need to resize
masks_np = [np.array(mask) for mask in masks]
hand_masks_np = []
obj_masks_np = []
for mask in masks_np:
    mask = cv2.resize(mask, (images_np[0].shape[1], images_np[0].shape[0]))
    hand_mask = np.zeros((mask.shape[0], mask.shape[1]))
    obj_mask = np.zeros((mask.shape[0], mask.shape[1]))
    hand_mask[mask[:, :, -1] == 255] = 255
    obj_mask[mask[:, :, 1] == 255] = 255
    hand_masks_np.append(hand_mask)
    obj_masks_np.append(obj_mask)

sample_folder = os.path.join("tmp_ho3d/", seq_name, exp_name)
os.makedirs(sample_folder, exist_ok=True)


# hand_detector = get_hand_bbox_detector()
# seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})
# hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
# obj_bboxes = [seq_boxes['objects']]
# # Initialize segmentation and hand pose estimation models
# mask_extractor = MaskExtractor(pointrend_model_weights="/remote-home/jiangshijian/homan/external/model_final_edd263.pkl") # detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl
# frankmocap_hand_checkpoint = "/remote-home/jiangshijian/homan/external/frankmocap/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
# hand_predictor = HandMocap(frankmocap_hand_checkpoint, "extra_data/smpl")

# Define camera parameters
height, width, _ = images_np[0].shape
image_size = max(height, width)
focal = 480
camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
camintrs = [camintr for _ in range(len(images_np))]

'''
obj_mask_infos: List
    full_mask: torch.bool H * W
'''
obj_mask_infos = process_masks_to_infos(obj_masks_np, hand_masks_np)
# person_parameters, _, _ = get_frame_infos(images_np,
#                                                     mask_extractor=mask_extractor,
#                                                     hand_predictor=hand_predictor,
#                                                     hand_bboxes=hand_bboxes,
#                                                     obj_bboxes=np.stack(obj_bboxes),
#                                                     sample_folder=sample_folder,
#                                                     camintr=camintrs,
#                                                     image_size=image_size,
#                                                     debug=False)
# for idx, (person_parameter, hand_mask) in enumerate(zip(person_parameters, hand_masks_np)):
#     hand_mask = (hand_mask == 255)
#     device = person_parameter['masks'].device
#     hand_mask = torch.from_numpy(hand_mask).unsqueeze(0).to(device)
#     person_parameters[idx]['masks'] = hand_mask

# frame_info = {"person_parameters": person_parameters[0], "obj_mask_infos": obj_mask_infos[0], "image": images_np[0]}
# super2d_img = viz_frame_info(frame_info,
#                                 sample_folder=sample_folder,
#                                 save=False)
# tmp_ = Image.fromarray(super2d_img)
# tmp_.save(os.path.join(sample_folder, 'tmp_2d.png'))

from homan.pose_optimization import find_optimal_poses


# ! Step1: Initialize object motion, based on the object masks
object_parameters = find_optimal_poses(
    images=images_np,
    image_size=images_np[0].shape,
    vertices=obj_verts_can,
    faces=obj_faces,
    annotations=obj_mask_infos,
    num_initializations=100,
    num_iterations=20, # Increase to get more accurate initializations
    Ks=np.stack(camintrs),
    viz_path=os.path.join(sample_folder, "optimal_pose.png"),
    debug=True,
)
'''
object_parameters: List, len(images)
    rotations: 1*3*3
    translations: 1*1*3

'''
# for person_param, obj_param, camintr in zip(person_parameters,
#                                         object_parameters,
#                                         camintrs):
    
#     maskutils.add_target_hand_occlusions(
#         person_param,
#         obj_param,
#         camintr,
#         debug=False,
#         sample_folder=sample_folder)
    
from homan.viz.colabutils import display_video
from homan.jointopt import optimize_hand_object, optimize_object

coarse_num_iterations = 201 # Increase to give more steps to converge
coarse_viz_step = 10 # Decrease to visualize more optimization steps

coarse_loss_weights = {
        "lw_inter": 0,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.0,
        "lw_contact": 0.0,
        "lw_scale_hand": 0.000,
        "lw_scale_obj": 0.000,
        "lw_v2d_hand": 00,
        "lw_smooth_hand": 0000,
        "lw_smooth_obj": 500,
        "lw_pca": 0.000,
    }

# Camera intrinsics in normalized coordinate
camintr_nc = np.stack(camintrs).copy().astype(np.float32)
camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

step2_folder = os.path.join(sample_folder, "jointoptim_step2")
step2_viz_folder = os.path.join(step2_folder, "viz")

# Coarse hand-object fitting
# model, loss_evolution, imgs = optimize_hand_object(
#     person_parameters=person_parameters,
#     object_parameters=object_parameters,
#     hand_proj_mode="persp",
#     objvertices=obj_verts_can,
#     objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
#     optimize_mano=False,
#     optimize_mano_beta=False,
#     optimize_object_scale=optim_obj_scale,
#     loss_weights=coarse_loss_weights,
#     image_size=image_size,
#     num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
#     images=np.stack(images_np),
#     camintr=camintr_nc,
#     state_dict=None,
#     viz_step=coarse_viz_step,
#     viz_folder=step2_viz_folder,
# )
model, loss_evolution, imgs = optimize_object(
    object_parameters=object_parameters,
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_object_scale=optim_obj_scale,
    loss_weights=coarse_loss_weights,
    image_size=image_size,
    num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=None,
    viz_step=coarse_viz_step,
    viz_folder=step2_viz_folder,
)
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
from homan.utils.geometry import rot6d_to_matrix
from tqdm import tqdm
obj_rot = rot6d_to_matrix(model.rotations_object).transpose(1, 2) # obj_coordinate to camera coordinate
obj_trans = model.translations_object

obj_rot_np = obj_rot.detach().cpu().numpy()
obj_trans_np = obj_trans.detach().cpu().numpy()
K = np.array([[focal, 0, width//2],
              [0, focal, height//2],
              [0, 0, 1]])
os.makedirs(os.path.join(sample_folder, "obj_infos"), exist_ok=True)
for i in tqdm(range(len(image_paths))):
    obj_mask = obj_mask_infos[i]['full_mask'].detach().cpu().numpy()
    data = {
        "obj_mask": obj_mask,
        "R": obj_rot_np[i],
        "T": obj_trans_np[i],
        "K": K,
        "obj_scale": model.int_scales_object.item()
    }
    path_id = image_paths[i].split("/")[-1][:-4]
    np.savez(os.path.join(sample_folder, "obj_infos/{}.npz".format(path_id)), **data)
