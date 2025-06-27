import os
import torch
import pybullet_data
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import math
import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

urdf = "kuka_iiwa/model.urdf"
search_path = pybullet_data.getDataPath()
full_urdf = os.path.join(search_path, urdf)
chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
d = "cpu"

chain = chain.to(device=d)
# paths to the link meshes are specified with their relative path inside the URDF
# we need to give them the path prefix as we need their absolute path to load
s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"))

th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d, requires_grad=False)
N = 10
th_perturbation = torch.randn(N - 1, 7, device=d) * 0.1
# N x 7 joint values
th = torch.cat((th.view(1, -1), th_perturbation + th))

global_pose = torch.zeros((N, 3), device=d)
mano_trans = torch.zeros((N, 3), device=d)
rot_matrix = pk.tf.euler_angles_to_matrix(mano_trans, "XYZ")
base_transform_matrix = torch.eye(4, device=d)
base_transform_matrix = base_transform_matrix.unsqueeze(0).repeat(N, 1, 1)
base_transform_matrix[..., :3, 3] = global_pose
base_transform_matrix[..., :3, :3] = rot_matrix
base_transform_matrix.requires_grad = True

tran = pk.tf.Transform3d(matrix=base_transform_matrix, device=d)
pnt = torch.rand((N, 100, 3), device=d)

pnt2 = tran.transform_points(pnt)
all_requires_grad = all(t.requires_grad for t in pnt2)
print("All tensors require grad:", all_requires_grad)

chain.set_base_transform(base_transform_matrix)
tg = chain.forward_kinematics(th)
m = tg.get_matrix()
pos = m[:, :3, 3]
pos.norm().backward()

y = 0.02
query_range = np.array([
    [-1, 0.5],
    [y, y],
    [-0.2, 0.8],
])
# M x 3 points
coords, pts = pv.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)
pts = torch.rand((N, 100, 3), device=d)
s.set_joint_configuration(th)
# N x M SDF value
# N x M x 3 SDF gradient
sdf_val, sdf_grad = s(pts)
print("SDF values shape:", sdf_val.shape)
print("SDF gradients shape:", sdf_grad.shape)