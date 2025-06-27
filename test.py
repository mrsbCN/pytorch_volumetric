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


# Setup optimization parameters
# batch_size is already defined (earlier)
obj_verts = torch.rand(batch_size, 500, 3, device=device) * 0.08
obj_cmap = torch.rand(batch_size, 500, device=device) > 0.7
obj_partition = torch.randint(0, 16, (batch_size, 500), device=device)
w_contact = 1.0

# Parameters for global pose
global_pose = torch.rand((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device, requires_grad=True) # This is a leaf
# For mano_trans, ensure it's a leaf tensor for the optimizer
mano_trans_params = torch.rand((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device, requires_grad=True) # This is a leaf

# Initial joint_config (will be made to require grad later for the second phase)
# n_joints is already defined from contact_hand_sdf.joint_names via the initial joint_config setting
joint_config_params = torch.zeros((batch_size, n_joints), device=contact_hand_sdf.device, dtype=contact_hand_sdf.dtype, requires_grad=False) # Starts as not trainable
contact_hand_sdf.set_joint_configuration(joint_config_params) # Set initial non-trainable joints

# Optimizer for the first phase (global pose)
optimizer_pose = torch.optim.Adam([global_pose, mano_trans_params], lr=0.01)

# First optimization loop (global pose)
print("Optimizing global pose...")
for it in range(2):
    optimizer_pose.zero_grad()

    rot_matrix = tf.euler_angles_to_matrix(mano_trans_params * 0.1, "XYZ") # Apply scaling here
    _base_transform_matrix = torch.eye(4, dtype=obj_verts.dtype, device=obj_verts.device).unsqueeze(0).repeat(batch_size, 1, 1)
    _base_transform_matrix[..., :3, 3] = global_pose
    _base_transform_matrix[..., :3, :3] = rot_matrix

    # ---- START DEBUG PRINTS for set_base_transform ----
    print(f"\nIter {it} - Before set_base_transform:")
    print(f"  _base_transform_matrix.requires_grad: {_base_transform_matrix.requires_grad}")
    print(f"  _base_transform_matrix.is_leaf: {_base_transform_matrix.is_leaf}")
    print(f"  _base_transform_matrix.grad_fn: {_base_transform_matrix.grad_fn}")
    # ---- END DEBUG PRINTS ----

    contact_hand_sdf.chain.set_base_transform(_base_transform_matrix)
    contact_hand_sdf.force_update_transforms() # Call to ensure SDF transforms are updated

    # ---- START DEBUG PRINTS for set_base_transform ----
    print(f"Iter {it} - After set_base_transform:")
    internal_base_transform_obj = contact_hand_sdf.chain.base_transform
    if internal_base_transform_obj is not None:
        internal_base_matrix = internal_base_transform_obj.get_matrix()
        print(f"  chain.base_transform.get_matrix().requires_grad: {internal_base_matrix.requires_grad}")
        print(f"  chain.base_transform.get_matrix().is_leaf: {internal_base_matrix.is_leaf}")
        print(f"  chain.base_transform.get_matrix().grad_fn: {internal_base_matrix.grad_fn}")
    else:
        print("  chain.base_transform is None")
    # ---- END DEBUG PRINTS ----

    # joint_config_params is already set in contact_hand_sdf and is not changing here

    pred, pred_p_full = contact_hand_sdf(obj_verts)

    # ---- START DEBUG PRINTS for pred_p_full ----
    print(f"\nIter {it} - After contact_hand_sdf(obj_verts):")
    print(f"  pred.requires_grad: {pred.requires_grad}")
    print(f"  pred_p_full.requires_grad: {pred_p_full.requires_grad}")
    if pred_p_full.grad_fn:
        print(f"  pred_p_full.grad_fn: {pred_p_full.grad_fn.name()}")
    else:
        print(f"  pred_p_full.grad_fn: None")
    # ---- END DEBUG PRINTS for pred_p_full ----

    pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
    loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
    total_loss = loss_contact
    
    # ---- START DEBUG PRINTS for total_loss ----
    print(f"\nIter {it} - Before total_loss.backward():")
    print(f"  total_loss requires_grad: {total_loss.requires_grad}")
    print(f"  total_loss is_leaf: {total_loss.is_leaf}")
    print(f"  total_loss grad_fn: {total_loss.grad_fn}")
    if total_loss.grad_fn and total_loss.grad_fn.next_functions:
        # Check the first operand's grad_fn name if it exists
        first_next_fn = total_loss.grad_fn.next_functions[0][0]
        if first_next_fn:
            print(f"  total_loss.grad_fn next_functions[0] name: {first_next_fn.name()}")
        else:
            print(f"  total_loss.grad_fn next_functions[0] is None")

    # ---- END DEBUG PRINTS for total_loss ----

    total_loss.backward()
    optimizer_pose.step()
    
    gp_grad_norm = global_pose.grad.norm().item() if global_pose.grad is not None else 'None'
    mt_grad_norm = mano_trans_params.grad.norm().item() if mano_trans_params.grad is not None else 'None'
    print(f"Iter {it}: Contact={loss_contact.item():.4f}, Global Pose Grad Norm: {gp_grad_norm}, Mano Trans Grad Norm: {mt_grad_norm}")

# Second optimization loop (global pose + joint_config)
print("\nOptimizing global pose and joint configuration...")
joint_config_params.requires_grad_(True) # Now make joints trainable
# Need to update contact_hand_sdf with the new requires_grad status if it caches it,
# but set_joint_configuration should be called if params change or their grad status changes affecting internal computations.
# For safety, we can call it again, though it might not be strictly necessary if only requires_grad changed on the tensor it already has.
contact_hand_sdf.set_joint_configuration(joint_config_params)


optimizer_combined = torch.optim.Adam([global_pose, mano_trans_params, joint_config_params], lr=0.01)

for it in range(2):
    optimizer_combined.zero_grad()

    rot_matrix = tf.euler_angles_to_matrix(mano_trans_params * 0.1, "XYZ") # Apply scaling here
    _base_transform_matrix = torch.eye(4, dtype=obj_verts.dtype, device=obj_verts.device).unsqueeze(0).repeat(batch_size, 1, 1)
    _base_transform_matrix[..., :3, 3] = global_pose
    _base_transform_matrix[..., :3, :3] = rot_matrix

    contact_hand_sdf.chain.set_base_transform(_base_transform_matrix)
    contact_hand_sdf.force_update_transforms() # Call to ensure SDF transforms are updated
    # joint_config_params is now trainable and set in contact_hand_sdf

    pred, pred_p_full = contact_hand_sdf(obj_verts)

    # ---- START DEBUG PRINTS for pred_p_full ----
    print(f"\nIter {it} - After contact_hand_sdf(obj_verts) - Second Loop:")
    print(f"  pred.requires_grad: {pred.requires_grad}")
    print(f"  pred_p_full.requires_grad: {pred_p_full.requires_grad}")
    if pred_p_full.grad_fn:
        print(f"  pred_p_full.grad_fn: {pred_p_full.grad_fn.name()}")
    else:
        print(f"  pred_p_full.grad_fn: None")
    # ---- END DEBUG PRINTS for pred_p_full ----

    pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
    loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
    total_loss = loss_contact
    
    # ---- START DEBUG PRINTS for total_loss ----
    print(f"\nIter {it} - Before total_loss.backward():")
    print(f"  total_loss requires_grad: {total_loss.requires_grad}")
    print(f"  total_loss is_leaf: {total_loss.is_leaf}")
    print(f"  total_loss grad_fn: {total_loss.grad_fn}")
    if total_loss.grad_fn and total_loss.grad_fn.next_functions:
        first_next_fn = total_loss.grad_fn.next_functions[0][0]
        if first_next_fn:
            print(f"  total_loss.grad_fn next_functions[0] name: {first_next_fn.name()}")
        else:
            print(f"  total_loss.grad_fn next_functions[0] is None")
    # ---- END DEBUG PRINTS for total_loss ----

    total_loss.backward()
    optimizer_combined.step()

    gp_grad_norm = global_pose.grad.norm().item() if global_pose.grad is not None else 'None'
    mt_grad_norm = mano_trans_params.grad.norm().item() if mano_trans_params.grad is not None else 'None'
    jc_grad_norm = joint_config_params.grad.norm().item() if joint_config_params.grad is not None else 'None'
    print(f"Iter {it}: Contact={loss_contact.item():.4f}, Global Pose Grad Norm: {gp_grad_norm}, Mano Trans Grad Norm: {mt_grad_norm}, Joint Config Grad Norm: {jc_grad_norm}")