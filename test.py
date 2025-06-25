"""
Test the enhanced ContactHandSDF implementation
"""

import torch
import pytorch_kinematics as pk
from pytorch_volumetric.contact_robot_hand_sdf import ContactHandSDF
import logging
import pytorch_kinematics.transforms as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load hand model
try:
    with open("hand/schunk_svh_hand_right.urdf", "rb") as f:
        chain = pk.build_mimic_chain_from_urdf(f.read())
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
n_joints = len(contact_hand_sdf.joint_names)
joint_config = torch.zeros(n_joints, device=device)
contact_hand_sdf.set_joint_configuration(joint_config)
print(f"✓ Joint configuration set ({n_joints} joints)")


# Setup optimization
batch_size = 10
optimizer = torch.optim.Adam([joint_config], lr=0.01)
obj_verts = torch.rand(batch_size, 500, 3, device=device) * 0.08
obj_cmap = torch.rand(batch_size, 500, device=device) > 0.7  # Binary contact
obj_partition = torch.randint(0, 16, (batch_size, 500), device=device)  # Random parts
w_contact = 1.0
global_pose = torch.zeros((3), dtype=obj_verts.dtype, device=obj_verts.device)
mano_trans = torch.zeros((3), dtype=obj_verts.dtype, device=obj_verts.device)
rot_matrix = tf.euler_angles_to_matrix(mano_trans, "XYZ")
base_transform_matrix = torch.zeros(( 4, 4), dtype=obj_verts.dtype, device=obj_verts.device)
base_transform_matrix[:3, 3] = global_pose
base_transform_matrix[:3, :3] = rot_matrix
base_transform_matrix.requires_grad = True

for it in range(1000):
    optimizer.zero_grad()

    contact_hand_sdf.chain.set_base_transform(base_transform_matrix)
    
    # Compute predictions with joint configuration (preserves gradient flow)
    pred, pred_p_full = contact_hand_sdf.forward(obj_verts, joint_config)
    
    # Contact loss (ContactGen style)
    pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
    loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
    
    # # Penetration loss
    # loss_pene = 0
    # if w_penetration > 0:
    #     mask = pred_p_full < eps
    #     masked_value = pred_p_full[mask]
    #     if len(masked_value) > 0:
    #         loss_pene = w_penetration * (-masked_value.sum()) / obj_verts.shape[0]
    
    # Total loss
    total_loss = loss_contact #+ loss_pene
    
    # Optimization step
    total_loss.backward()
    optimizer.step()
    
    if it % 100 == 0:
        print(f"Iter {it}: Contact={loss_contact:.4f}") #, Penetration={loss_pene:.4f}")
