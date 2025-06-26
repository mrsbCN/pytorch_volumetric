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
    try:
        chain = pk.build_chain_from_urdf(urdf_string)
    except Exception as e:
        print(f"Error building chain from URDF: {e}")
        print("Ensure URDF is correct, including effort/velocity in joint limits.")
        return

    chain = chain.to(dtype=dtype, device=device)
    tool_frame_name = "tool_tip"
    print(f"Kinematic chain built for {chain._root.name}. Tool frame: {tool_frame_name}")


    # Test Case 1: No Base Transform (Should be Identity)
    print("\n--- Test Case 1: No Base Transform (Identity) ---")
    joint_angle_val = math.pi / 2  # 90 degrees
    th0 = torch.tensor([[joint_angle_val]], dtype=dtype, device=device) # Shape (1,1) for single joint

    # Reset to default base transform (identity) for this test
    default_base_transform_matrix = torch.eye(4, dtype=dtype, device=device)
    chain.set_base_transform(default_base_transform_matrix)

    fk_results_no_base = chain.forward_kinematics(th0)
    tool_transform_no_base_obj = fk_results_no_base[tool_frame_name]
    tool_transform_no_base = tool_transform_no_base_obj.get_matrix()

    expected_pos_no_base = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    actual_pos_no_base = tool_transform_no_base[0, :3, 3]
    print(f"Expected position (no base transform): {expected_pos_no_base.cpu().numpy()}")
    print(f"Actual position (no base transform): {actual_pos_no_base.cpu().numpy()}")
    assert torch.allclose(actual_pos_no_base, expected_pos_no_base, atol=1e-5), "Test Case 1 Failed!"
    print("Test Case 1 Passed!")

    # Test Case 2: With Base Translation
    print("\n--- Test Case 2: With Base Translation ---")
    base_translation = torch.tensor([10.0, 20.0, 30.0], dtype=dtype, device=device)

    base_transform_matrix_trans = torch.eye(4, dtype=dtype, device=device)
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
    angle_base_z = math.pi / 2
    c, s = math.cos(angle_base_z), math.sin(angle_base_z)
    base_rot_matrix_z90 = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype, device=device)

    base_transform_matrix_rot = torch.eye(4, dtype=dtype, device=device)
    base_transform_matrix_rot[:3, :3] = base_rot_matrix_z90

    chain.set_base_transform(base_transform_matrix_rot)

    fk_results_with_rot = chain.forward_kinematics(th0)
    tool_transform_with_rot_obj = fk_results_with_rot[tool_frame_name]
    tool_transform_with_rot = tool_transform_with_rot_obj.get_matrix()

    expected_pos_with_rot = torch.matmul(base_rot_matrix_z90, expected_pos_no_base.reshape(3,1)).squeeze()
    actual_pos_with_rot = tool_transform_with_rot[0, :3, 3]
    print(f"Expected position (with base rotation): {expected_pos_with_rot.cpu().numpy()}")
    print(f"Actual position (with base rotation): {actual_pos_with_rot.cpu().numpy()}")
    assert torch.allclose(actual_pos_with_rot, expected_pos_with_rot, atol=1e-5), "Test Case 3 Failed!"
    print("Test Case 3 Passed!")

    # Test Case 4: Gradient Flow to Base Transform
    print("\n--- Test Case 4: Gradient Flow to Base Transform ---")
    base_trans_grad_test_matrix = torch.eye(4, dtype=dtype, device=device)
    base_trans_grad_test_matrix.requires_grad_(True)

    chain.set_base_transform(base_trans_grad_test_matrix)
    th0_no_grad = th0.detach()

    fk_results_grad = chain.forward_kinematics(th0_no_grad)
    tool_transform_grad_obj = fk_results_grad[tool_frame_name]
    tool_transform_grad = tool_transform_grad_obj.get_matrix()
    loss = tool_transform_grad[0, :3, 3].sum()
    loss.backward()

    assert base_trans_grad_test_matrix.grad is not None, "Test Case 4 Failed: Grad is None for base_transform_matrix!"
    print(f"Gradient for base_transform_matrix:\n{base_trans_grad_test_matrix.grad}")
    assert base_trans_grad_test_matrix.grad.norm() > 0, "Test Case 4 Failed: Grad norm is zero for base_transform_matrix!"
    expected_grad_elements = [(0,1), (1,1), (2,1), (0,3), (1,3), (2,3)]
    for r,c in expected_grad_elements:
        assert abs(base_trans_grad_test_matrix.grad[r,c].item() - 1.0) < 1e-5, \
            f"Test Case 4 Failed: Expected grad approx 1.0 at ({r},{c}), got {base_trans_grad_test_matrix.grad[r,c].item()}"
    print("Test Case 4 Passed!")

    # --- New Test Cases for Batching ---

    # Test Case 5: Single th, Batched base_transform
    print("\n--- Test Case 5: Single th, Batched base_transform ---")
    th_single = th0.clone() # Shape (1,1)

    base_trans_1_mat = torch.eye(4, dtype=dtype, device=device)
    base_trans_1_mat[:3, 3] = torch.tensor([1., 2., 3.], dtype=dtype, device=device)

    angle_base_z_tc5 = math.pi / 4 # 45 deg
    c5, s5 = math.cos(angle_base_z_tc5), math.sin(angle_base_z_tc5)
    base_rot_2_mat_tc5 = torch.tensor([[c5, -s5, 0], [s5, c5, 0], [0, 0, 1]], dtype=dtype, device=device)
    base_trans_2_mat = torch.eye(4, dtype=dtype, device=device)
    base_trans_2_mat[:3, :3] = base_rot_2_mat_tc5
    base_trans_2_mat[:3, 3] = torch.tensor([4., 5., 6.], dtype=dtype, device=device)

    batched_base_matrices = torch.stack([base_trans_1_mat, base_trans_2_mat], dim=0) # Shape (2, 4, 4)
    chain.set_base_transform(batched_base_matrices)

    fk_results_tc5 = chain.forward_kinematics(th_single)
    tool_transform_tc5 = fk_results_tc5[tool_frame_name].get_matrix()

    assert tool_transform_tc5.shape == (2, 4, 4), \
        f"Test Case 5 Failed: Expected shape (2,4,4), got {tool_transform_tc5.shape}"

    # Expected for item 0 (base_trans_1_mat)
    expected_pos_5_0 = expected_pos_no_base + base_trans_1_mat[:3,3]
    actual_pos_5_0 = tool_transform_tc5[0, :3, 3]
    assert torch.allclose(actual_pos_5_0, expected_pos_5_0, atol=1e-5), \
        f"Test Case 5 Failed (Item 0): Expected {expected_pos_5_0}, Got {actual_pos_5_0}"

    # Expected for item 1 (base_trans_2_mat)
    pos_in_rotated_base_5_1 = torch.matmul(base_trans_2_mat[:3, :3], expected_pos_no_base.reshape(3,1)).squeeze()
    expected_pos_5_1 = pos_in_rotated_base_5_1 + base_trans_2_mat[:3,3]
    actual_pos_5_1 = tool_transform_tc5[1, :3, 3]
    assert torch.allclose(actual_pos_5_1, expected_pos_5_1, atol=1e-5), \
        f"Test Case 5 Failed (Item 1): Expected {expected_pos_5_1}, Got {actual_pos_5_1}"
    print("Test Case 5 Passed!")

    # Test Case 6: Batched th, Batched base_transform (Matching Batch Sizes)
    print("\n--- Test Case 6: Batched th, Batched base_transform (Matching Batch Sizes) ---")
    th_batch_tc6 = torch.tensor([[math.pi/2], [0.0]], dtype=dtype, device=device) # Shape (2,1)
    # batched_base_matrices (shape (2,4,4)) is already set on the chain from TC5

    fk_results_tc6 = chain.forward_kinematics(th_batch_tc6)
    tool_transform_tc6 = fk_results_tc6[tool_frame_name].get_matrix()

    assert tool_transform_tc6.shape == (2, 4, 4), \
        f"Test Case 6 Failed: Expected shape (2,4,4), got {tool_transform_tc6.shape}"

    # Item 0: th=pi/2, base=base_trans_1_mat
    # expected_pos_no_base (0,1,0) is for th=pi/2
    expected_pos_6_0 = expected_pos_no_base + base_trans_1_mat[:3,3]
    actual_pos_6_0 = tool_transform_tc6[0, :3, 3]
    assert torch.allclose(actual_pos_6_0, expected_pos_6_0, atol=1e-5), \
        f"Test Case 6 Failed (Item 0): Expected {expected_pos_6_0}, Got {actual_pos_6_0}"

    # Item 1: th=0.0, base=base_trans_2_mat
    # For th=0.0, link1 R is Identity, so tool_tip in base frame is (1,0,0)
    pos_in_base_frame_th_zero = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    pos_in_rotated_base_6_1 = torch.matmul(base_trans_2_mat[:3, :3], pos_in_base_frame_th_zero.reshape(3,1)).squeeze()
    expected_pos_6_1 = pos_in_rotated_base_6_1 + base_trans_2_mat[:3,3]
    actual_pos_6_1 = tool_transform_tc6[1, :3, 3]
    assert torch.allclose(actual_pos_6_1, expected_pos_6_1, atol=1e-5), \
        f"Test Case 6 Failed (Item 1): Expected {expected_pos_6_1}, Got {actual_pos_6_1}"
    print("Test Case 6 Passed!")

    # Test Case 7: Gradient Flow with Batched base_transform
    print("\n--- Test Case 7: Gradient Flow with Batched base_transform ---")
    th_grad_test_tc7 = th0.clone().detach() # Single th (1,1), no grad

    # Use batched_base_matrices from TC5, ensure it requires grad for this test
    base_matrices_grad_test_tc7 = batched_base_matrices.clone().requires_grad_(True)
    chain.set_base_transform(base_matrices_grad_test_tc7)

    fk_results_tc7 = chain.forward_kinematics(th_grad_test_tc7) # th is (1,1), base is (2,4,4) -> output should be (2,4,4)
    tool_transform_tc7 = fk_results_tc7[tool_frame_name].get_matrix()

    assert tool_transform_tc7.shape == (2,4,4), f"Test Case 7: Output shape mismatch, got {tool_transform_tc7.shape}"

    loss_tc7 = tool_transform_tc7[:, :3, 3].sum() # Sum all x,y,z for both batch items
    loss_tc7.backward()

    assert base_matrices_grad_test_tc7.grad is not None, \
        "Test Case 7 Failed: Grad is None for batched base_transform_matrix!"
    assert base_matrices_grad_test_tc7.grad.norm() > 0, \
        "Test Case 7 Failed: Grad norm is zero for batched base_transform_matrix!"
    print(f"Gradient norm for batched base_transform_matrix: {base_matrices_grad_test_tc7.grad.norm().item()}")
    print("Test Case 7 Passed!")

    # Test Case 8: Incompatible Batch Sizes (Error Check)
    print("\n--- Test Case 8: Incompatible Batch Sizes (Error Check) ---")
    # chain still has batched_base_matrices (B_base = 2) from TC7
    th_batch_incompatible = torch.rand(3, chain.n_joints, dtype=dtype, device=device) # B_th = 3

    try:
        chain.forward_kinematics(th_batch_incompatible)
        # Should not reach here
        assert False, "Test Case 8 Failed: ValueError for incompatible batch sizes not raised."
    except ValueError as e:
        print(f"Correctly caught ValueError: {e}")
        print("Test Case 8 Passed!")
    except Exception as e:
        assert False, f"Test Case 8 Failed: Incorrect exception type raised: {type(e)} {e}"


    print("\nAll test_base_transform.py tests (including batching) passed!")

if __name__ == "__main__":
    run_tests()