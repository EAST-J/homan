import numpy as np
from renderer import Visualizer
import cv2
from PIL import Image
import trimesh
import os
from glob import glob
from tqdm import tqdm
import pickle as pkl

seq_name = "SB14"
exp_name ="pred"
data_root = "/remote-home/jiangshijian/data/HO3D_v3/train"
start_idx = 900
end_idx = 1100
image_paths = sorted(glob(os.path.join(data_root, seq_name, "rgb", "*.jpg")))[start_idx:end_idx]
sample_folder = os.path.join("tmp_ho3d/", seq_name, exp_name)
print(image_paths)
print(len(image_paths))
with open(os.path.join(data_root, seq_name, "meta", "0000.pkl"), "rb") as f:
    data = pkl.load(f, encoding='latin1')
objName = data["objName"]
if exp_name == 'gt':
    obj_path = os.path.join("local_data/datasets/ycbmodels/", objName, "textured_simple_2000.obj")
    obj_scale = 0.08
else:
    obj_path = "/remote-home/jiangshijian/shap-e/cleaner.ply"
    obj_scale = 0.1
obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
obj_faces = np.array(obj_mesh.faces)
# Convert images to numpy 
images = [Image.open(image_path) for image_path in image_paths]
images_np = [np.array(image) for image in images]
focal = 480
height, width, _ = images_np[0].shape
vis = Visualizer((height, width))
os.makedirs(os.path.join(sample_folder, "render_res"), exist_ok=True)
for i in tqdm(range(len(image_paths))):
    id = image_paths[i].split("/")[-1][:-4]
    obj_info = np.load(os.path.join(sample_folder, "obj_infos/{}.npz").format(id))
    R = obj_info["R"]
    T = obj_info["T"]
    obj_scale = obj_info["obj_scale"]
    obj_v_trans = (obj_scale * obj_verts_can) @ R.T + T
    res_img = vis.draw_mesh(images_np[i] / 255., obj_v_trans, obj_mesh.faces, (focal, focal, width // 2, height // 2))
    cv2.imwrite(os.path.join(sample_folder, 'render_res/{}.jpg'.format(id)), res_img[:, :, ::-1] * 255)