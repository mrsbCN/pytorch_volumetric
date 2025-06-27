"""
Test the enhanced ContactHandSDF implementation
"""

import torch
import pytorch_kinematics as pk
from pytorch_volumetric.contact_robot_hand_sdf import ContactHandSDF
import logging
import pytorch_kinematics.transforms as tf
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load hand model
try:
    with open("hand/schunk_svh_hand_right.urdf", "rb") as f:
        chain = pk.build_mimic_chain_from_urdf(f.read())
    device = "cpu"
    chain = chain.to(device=device)
    print(f"✓ Hand model loaded on {device}")
except Exception as e:
    print(f"✗ Failed to load hand model: {e}")

# Create ContactHandSDF
try:
    contact_hand_sdf = ContactHandSDF(chain, path_prefix="hand/")
    print("✓ ContactHandSDF created successfully")
except Exception as e:
    print(f"✗ Failed to create ContactHandSDF: {e}")

# Set initial joint configuration
batch_size = 10  # Define batch_size earlier for joint_config
n_joints = len(contact_hand_sdf.joint_names)
# Create a batched joint_config, e.g., all zeros
joint_config = torch.zeros(batch_size, n_joints, device=device, requires_grad=True) 
contact_hand_sdf.set_joint_configuration(joint_config)
print(f"✓ Batched joint configuration set ({batch_size} batches, {n_joints} joints per batch)")


# Setup optimization
# batch_size is already defined
optimizer = torch.optim.Adam([joint_config], lr=0.01) # joint_config is now the learnable batched parameter
obj_verts = torch.rand(batch_size, 500, 3, device=device) * 0.08
obj_cmap = torch.rand(batch_size, 500, device=device) > 0.7  # Binary contact
obj_partition = torch.randint(0, 16, (batch_size, 500), device=device)  # Random parts
w_contact = 1.0
global_pose = torch.rand((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device,requires_grad=True)
mano_trans = torch.rand((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device,requires_grad=True) * 0.1  # Random rotation angles
rot_matrix = tf.euler_angles_to_matrix(mano_trans, "XYZ")
base_transform_matrix = torch.eye(4, dtype=obj_verts.dtype, device=obj_verts.device,requires_grad=True)
base_transform_matrix = base_transform_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
base_transform_matrix[..., :3, 3] = global_pose
base_transform_matrix[..., :3, :3] = rot_matrix
M = len(contact_hand_sdf.joint_names)
joint_config = torch.zeros((batch_size, M), device=contact_hand_sdf.device, dtype=contact_hand_sdf.dtype, requires_grad=False)

for it in range(2):
    optimizer.zero_grad()

    contact_hand_sdf.chain.set_base_transform(base_transform_matrix)

    contact_hand_sdf.set_joint_configuration(joint_config)

    # Compute predictions with joint configuration (preserves gradient flow)
    pred, pred_p_full = contact_hand_sdf(obj_verts)
    
    # Contact loss (ContactGen style)
    pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
    loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        
    # Total loss
    total_loss = loss_contact #+ loss_pene
    
    # Optimization step
    total_loss.backward()
    optimizer.step()
    
    print(f"Iter {it}: Contact={loss_contact:.4f}") #, Penetration={loss_pene:.4f}")

joint_config.requires_grad = True  # Ensure joint_config is a learnable parameter
for it in range(2):
    optimizer.zero_grad()

    contact_hand_sdf.chain.set_base_transform(base_transform_matrix)

    contact_hand_sdf.set_joint_configuration(joint_config)

    # Compute predictions with joint configuration (preserves gradient flow)
    pred, pred_p_full = contact_hand_sdf(obj_verts)
    
    # Contact loss (ContactGen style)
    pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
    loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        
    # Total loss
    total_loss = loss_contact #+ loss_pene
    
    # Optimization step
    total_loss.backward()
    optimizer.step()
    print(f"Iter {it}: Contact={loss_contact:.4f}") #, Penetration={loss_pene:.4f}")