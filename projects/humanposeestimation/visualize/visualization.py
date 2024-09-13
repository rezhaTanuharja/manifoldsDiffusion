import torch
import numpy as np
#
from human_body_prior.tools.omni_tools import copy2cpu as c2c
# from os import path as osp

#TODO: this script needs to be completely cleaned up

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# amass_npz_fname = './downloads/dmpl_sample.npz'
# amass_npz_fname = './downloads/processed_A2.npz'
# amass_npz_fname = './downloads/postprocessed_A2.npz'
# amass_npz_fname = './projects/humanposeestimation/visualize/data/A2_subject1_noised.npz'
amass_npz_fname = "./extractedData/ACCAD/Female1Walking_c3d/B1 - stand to walk_poses.npz"
bdata = np.load(amass_npz_fname)

# subject_gender = bdata['gender']

# print(f'The subject of the mocap sequence is {subject_gender}')

from human_body_prior.body_model.body_model import BodyModel

bm_fname = f'./downloads/body_models/smplh/male/model.npz'
dmpl_fname = f'./downloads/body_models/dmpls/male/model.npz'

num_betas = 16
num_dmpls = 8

bm = BodyModel(
    bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname
).to(comp_device)
faces = c2c(bm.f)

# time_length = len(bdata['trans'])
time_length = 5

copied_data = bdata['poses']

copied_data[400, 9:66] = copied_data[200, 9:66]
copied_data[400, 3:6] = copied_data[200, 3:6]
copied_data[0, 9:66] = copied_data[200, 9:66]
# copied_data[0, 3:6] = copied_data[200, 3:6]

body_parms = {
    'root_orient': torch.Tensor(copied_data[:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(copied_data[:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(copied_data[:, 66:]).to(comp_device), # controls the finger articulation
    # 'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    # 'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    # 'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

# body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas']})
#
# def vis_body_pose_beta(fId = 0):
#     body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
#     mv.set_static_meshes([body_mesh])
#     body_image = mv.render(render_wireframe=False)
#     # print(body_image)
#     show_image(body_image)

# vis_body_pose_beta(fId=1)

# body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})
body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'pose_hand']})

def vis_body_pose_hand(fId = 0, i = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image, i)

for i in range(bdata['poses'].shape[0]):
    if i % 50 == 0:
        vis_body_pose_hand(fId=i, i = i)
