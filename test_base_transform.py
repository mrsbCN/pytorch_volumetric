import torch
import pytorch_kinematics as pk
from pytorch_kinematics import transforms as tf # For tf.Transform3D
import math # For pi
import os

# Ensure pytorch_kinematics is installed in editable mode:
# pip install -e ./pytorch_kinematics (if in root and pytorch_kinematics is a subdir)
# or cd pytorch_kinematics; pip install -e .

def run_tests():
    # Device and Dtype Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32

    # Define URDF for a Simple Arm
    urdf_string = """
    <robot name="single_link_arm">
        <link name="base_link"/>
        <link name="link1">
            <visual><geometry><box size="1.0 0.1 0.1"/></geometry></visual>
        </link>
        <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0 0 0" rpy="0 0 0"/> <!-- Joint at base origin -->
            <axis xyz="0 0 1"/> <!-- Rotate around Z -->
            <limit lower="-3.14" upper="3.14" effort="1.0" velocity="1.0"/>
        </joint>
        <link name="tool_tip">
             <visual><geometry><sphere radius="0.01"/></geometry></visual>
        </link>
        <joint name="tool_joint" type="fixed">
            <parent link="link1"/>
            <child link="tool_tip"/>
            <origin xyz="1.0 0 0" rpy="0 0 0"/> <!-- Tip at the end of link1 -->
        </joint>
    </robot>
    """
    # The URDF parser in pytorch_kinematics might not use root_link_name from build_chain_from_urdf,
    # it typically infers the root or uses the first link if not specified.
    # For this URDF, base_link is the natural root.
    try:
        chain = pk.build_chain_from_urdf(urdf_string)
    except Exception as e:
        print(f"Error building chain from URDF: {e}")
        # Adding effort and velocity to limit tags if that was the issue,
        # which should have been fixed in the URDF string already.
        print("Ensure URDF is correct, including effort/velocity in joint limits.")
        return

    chain = chain.to(dtype=dtype, device=device)
    tool_frame_name = "tool_tip"
    print(f"Kinematic chain built for {chain._root.name}. Tool frame: {tool_frame_name}")


    # Test Case 1: No Base Transform (Should be Identity)
    print("\n--- Test Case 1: No Base Transform (Identity) ---")
    joint_angle_val = math.pi / 2  # 90 degrees
    th0 = torch.tensor([[joint_angle_val]], dtype=dtype, device=device)

    fk_results_no_base = chain.forward_kinematics(th0)
    tool_transform_no_base_obj = fk_results_no_base[tool_frame_name]
    tool_transform_no_base = tool_transform_no_base_obj.get_matrix()

    # Analytical calculation for tool_tip relative to base:
    # Link1 length is 1.0. Rotated 90 deg around Z.
    # Origin of joint1 is (0,0,0). Axis is (0,0,1).
    # Transformation for joint1:
    # R = [[cos(pi/2), -sin(pi/2), 0], [sin(pi/2), cos(pi/2), 0], [0, 0, 1]] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # t = [0, 0, 0]'
    # T_link1_in_base = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # Tool_tip is fixed at (1,0,0) in link1's frame. T_tooltip_in_link1 = Identity with translation (1,0,0)
    # T_tooltip_in_base = T_link1_in_base @ T_tooltip_in_link1
    # Position of tool_tip in base frame = R_link1_in_base * [1,0,0]' + t_link1_in_base = [0, 1, 0]'
    expected_pos_no_base = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    actual_pos_no_base = tool_transform_no_base[0, :3, 3]
    print(f"Expected position (no base transform): {expected_pos_no_base.cpu().numpy()}")
    print(f"Actual position (no base transform): {actual_pos_no_base.cpu().numpy()}")
    assert torch.allclose(actual_pos_no_base, expected_pos_no_base, atol=1e-5), "Test Case 1 Failed!"
    print("Test Case 1 Passed!")

    # Test Case 2: With Base Translation
    print("\n--- Test Case 2: With Base Translation ---")
    base_translation = torch.tensor([10.0, 20.0, 30.0], dtype=dtype, device=device)
    # base_rot_matrix = torch.eye(3, dtype=dtype, device=device) # Identity rotation

    base_transform_matrix_trans = torch.eye(4, dtype=dtype, device=device)
    # base_transform_matrix_trans[:3, :3] = base_rot_matrix # Already identity
    base_transform_matrix_trans[:3, 3] = base_translation

    chain.set_base_transform(base_transform_matrix_trans)

    fk_results_with_trans = chain.forward_kinematics(th0)
    tool_transform_with_trans_obj = fk_results_with_trans[tool_frame_name]
    tool_transform_with_trans = tool_transform_with_trans_obj.get_matrix()

    expected_pos_with_trans = expected_pos_no_base + base_translation
    actual_pos_with_trans = tool_transform_with_trans[0, :3, 3]
    print(f"Expected position (with base translation): {expected_pos_with_trans.cpu().numpy()}")
    print(f"Actual position (with base translation): {actual_pos_with_trans.cpu().numpy()}")
    assert torch.allclose(actual_pos_with_trans, expected_pos_with_trans, atol=1e-5), "Test Case 2 Failed!"
    print("Test Case 2 Passed!")

    # Test Case 3: With Base Rotation (90 deg around world Z-axis)
    print("\n--- Test Case 3: With Base Rotation ---")
    # Original tool pos in base frame (from Test Case 1): (0, 1, 0)
    # World R_base (rotation of base frame in world) = 90 deg around Z
    # R_world_base = [[cos(pi/2), -sin(pi/2), 0], [sin(pi/2), cos(pi/2), 0], [0, 0, 1]]
    #              = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # New world pos = R_world_base * original_pos_in_base_frame (which is expected_pos_no_base)
    #               = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] * [0, 1, 0]' = [-1, 0, 0]'

    angle_base_z = math.pi / 2
    c, s = math.cos(angle_base_z), math.sin(angle_base_z)
    base_rot_matrix_z90 = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype, device=device)

    base_transform_matrix_rot = torch.eye(4, dtype=dtype, device=device)
    base_transform_matrix_rot[:3, :3] = base_rot_matrix_z90
    # No translation for this part, so base_transform_matrix_rot[:3, 3] remains [0,0,0]

    chain.set_base_transform(base_transform_matrix_rot)

    fk_results_with_rot = chain.forward_kinematics(th0)
    tool_transform_with_rot_obj = fk_results_with_rot[tool_frame_name]
    tool_transform_with_rot = tool_transform_with_rot_obj.get_matrix()

    # Calculate expected position: R_world_base @ expected_pos_no_base
    # Ensure expected_pos_no_base is (3,1) for matmul or handle broadcasting
    expected_pos_with_rot = torch.matmul(base_rot_matrix_z90, expected_pos_no_base.reshape(3,1)).squeeze()
    actual_pos_with_rot = tool_transform_with_rot[0, :3, 3]
    print(f"Expected position (with base rotation): {expected_pos_with_rot.cpu().numpy()}")
    print(f"Actual position (with base rotation): {actual_pos_with_rot.cpu().numpy()}")
    assert torch.allclose(actual_pos_with_rot, expected_pos_with_rot, atol=1e-5), "Test Case 3 Failed!"
    print("Test Case 3 Passed!")

    # Test Case 4: Gradient Flow to Base Transform
    print("\n--- Test Case 4: Gradient Flow to Base Transform ---")
    # Reset base transform to identity, but requiring grad
    # Must use a new tensor for requires_grad to take effect if it was previously set by set_base_transform
    # with a non-leaf tensor.
    base_trans_grad_test_matrix = torch.eye(4, dtype=dtype, device=device)
    base_trans_grad_test_matrix.requires_grad_(True) # Set requires_grad in place

    chain.set_base_transform(base_trans_grad_test_matrix)

    # Ensure th0 does not require grad for this specific test, to isolate grad to base_transform
    th0_no_grad = th0.detach()

    fk_results_grad = chain.forward_kinematics(th0_no_grad)
    tool_transform_grad_obj = fk_results_grad[tool_frame_name]
    tool_transform_grad = tool_transform_grad_obj.get_matrix() # This is T_world_tool

    # Dummy loss on the output position
    # The position of the tool tip in world frame is tool_transform_grad[0, :3, 3]
    # Position in base frame was (0,1,0). If base_transform is Identity, world pos is (0,1,0).
    # Sum of world pos = 1.0
    loss = tool_transform_grad[0, :3, 3].sum()
    loss.backward()

    assert base_trans_grad_test_matrix.grad is not None, "Test Case 4 Failed: Grad is None for base_transform_matrix!"
    # For an identity base_transform, and loss = x+y+z of the tool's world position.
    # Tool's position in its own base frame (after joint rotation) is P_base = (0, 1, 0).
    # World position P_world = R_base * P_base + t_base.
    # If base_transform is T_base = [[R_base, t_base], [0, 1]],
    # P_world_homogeneous = T_base @ [P_base_homogeneous]
    # P_world_x = R_base[0,0]*P_base_x + R_base[0,1]*P_base_y + R_base[0,2]*P_base_z + t_base_x
    # P_world_y = R_base[1,0]*P_base_x + R_base[1,1]*P_base_y + R_base[1,2]*P_base_z + t_base_y
    # P_world_z = R_base[2,0]*P_base_x + R_base[2,1]*P_base_y + R_base[2,2]*P_base_z + t_base_z
    # Loss = P_world_x + P_world_y + P_world_z.
    # d(Loss)/d(t_base_x) = 1. d(Loss)/d(t_base_y) = 1. d(Loss)/d(t_base_z) = 1.
    # d(Loss)/d(R_base[0,1]) = P_base_y = 1. Other d(Loss)/d(R_base_ij) involving P_base_y should be 1.
    # So, grad matrix should have 1s in the translation part and for R[0,1], R[1,1], R[2,1] if P_base_y=1.
    # Specifically, for P_base = [0,1,0]:
    # dL/dt_x=1, dL/dt_y=1, dL/dt_z=1
    # dL/dR_01=1 (coeff of P_base_y in P_world_x)
    # dL/dR_11=1 (coeff of P_base_y in P_world_y)
    # dL/dR_21=1 (coeff of P_base_y in P_world_z)
    # All other dL/dR_ij = 0
    # So grad matrix should have 1s at indices (0,3), (1,3), (2,3) and (0,1), (1,1), (2,1)

    print(f"Gradient for base_transform_matrix:\n{base_trans_grad_test_matrix.grad}")
    assert base_trans_grad_test_matrix.grad.norm() > 0, "Test Case 4 Failed: Grad norm is zero for base_transform_matrix!"
    # Check specific gradient components
    expected_grad_elements = [(0,1), (1,1), (2,1), (0,3), (1,3), (2,3)]
    for r,c in expected_grad_elements:
        assert abs(base_trans_grad_test_matrix.grad[r,c].item() - 1.0) < 1e-5, \
            f"Test Case 4 Failed: Expected grad approx 1.0 at ({r},{c}), got {base_trans_grad_test_matrix.grad[r,c].item()}"

    print("Test Case 4 Passed!")

    print("\nAll test_base_transform.py tests passed!")

if __name__ == "__main__":
    run_tests()
