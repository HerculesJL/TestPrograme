import torch
import numpy as np
import nksr
from pycg import vis, exp
from pathlib import Path

def load_bunny_example():
    bunny_path =  "1.ply"
    bunny_geom = vis.from_file(bunny_path)
    return bunny_geom

device = torch.device("cuda:0")
bunny_geom = load_bunny_example()

input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

reconstructor = nksr.Reconstructor(device)
field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
mesh = field.extract_dual_mesh(mise_iter=1)
mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)

vis.show_3d([mesh], [test_geom])