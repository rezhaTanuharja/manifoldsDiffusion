import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
#
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image

import matplotlib.pyplot as plt

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname=  './extractedData/ACCAD/Male2Running_c3d/C11 - run turn left 90_poses.npz'
bdata = np.load(amass_npz_fname)

bm_fname = f'./downloads/body_models/smplh/male/model.npz'
dmpl_fname = f'./downloads/body_models/dmpls/male/model.npz'

num_betas = 16
num_dmpls = 8

bm = BodyModel(
    bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname
).to(comp_device)
faces = c2c(bm.f)

copied_data = bdata['poses']

body_parms = {
    'root_orient': torch.Tensor(copied_data[:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(copied_data[:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(copied_data[:, 66:]).to(comp_device), # controls the finger articulation
    # 'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    # 'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    # 'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}


imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'pose_hand']})
#
def vis_body_pose_hand(fId = 0, i = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)
    plt.savefig(f'./downloads/normal/mr_0_{i}.png', format = 'png', transparent=True)
    plt.savefig(f'./downloads/normal/mr_0_{i}.pdf', format = 'pdf', transparent=True)
    plt.close()

for i in range(bdata['poses'].shape[0]):
    if i % 50 == 0:
        vis_body_pose_hand(fId=i, i = i)
