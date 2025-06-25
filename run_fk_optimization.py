import torch
import pytorch_kinematics as pk
import os
import math # For pi

# IMPORTANT SETUP INSTRUCTIONS:
# Before running this script, ensure you have installed the modified pytorch_kinematics:
# 1. Navigate to the `pytorch_kinematics` directory in your terminal.
# 2. Run: pip install -e .
# This will install the library in editable mode, so your changes are used.

def run_fk_opt():
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32

    # Define URDF for a Simple Arm
    urdf_string = """
    <robot name="simple_3_link_arm">
        <link name="base_link"/>
        <link name="link1">
            <visual><geometry><box size="0.5 0.1 0.1"/></geometry></visual>
        </link>
        <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="1.0" velocity="1.0"/>
        </joint>
        <link name="link2">
            <visual><geometry><box size="0.5 0.1 0.1"/></geometry></visual>
        </link>
        <joint name="joint2" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <origin xyz="0.5 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="1.0" velocity="1.0"/>
        </joint>
        <link name="end_effector_link">
            <visual><geometry><sphere radius="0.05"/></geometry></visual>
        </link>
        <joint name="joint_ee" type="fixed">
            <parent link="link2"/>
            <child link="end_effector_link"/>
            <origin xyz="0.5 0 0" rpy="0 0 0"/>
        </joint>
    </robot>
    """

    # Build Kinematic Chain
    # Note: Removed root_link_name as it caused issues previously and is not strictly needed for this URDF.
    # The parser should correctly identify 'base_link' as the root.
    try:
        chain = pk.build_chain_from_urdf(urdf_string)
    except Exception as e:
        print(f"Error building chain from URDF: {e}")
        print("Ensure your URDF is correct and all necessary attributes (like effort and velocity for limits) are set.")
        return

    chain = chain.to(dtype=dtype, device=device)
    end_effector_name = "end_effector_link"
    print(f"Kinematic chain created. End effector: {end_effector_name}")

    # Define and set a base transformation for the robot
    # For example, move the robot's base by (0.2, 0.1, 0.0) and rotate 45 deg around Z
    base_offset_translation = torch.tensor([0.2, 0.1, 0.0], dtype=dtype, device=device)
    angle_rad = math.pi / 4  # 45 degrees
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    base_offset_rotation_matrix = torch.tensor([[c, -s, 0],
                                                [s,  c, 0],
                                                [0,  0, 1]], dtype=dtype, device=device)

    # Create the 4x4 base transformation matrix
    world_H_base = torch.eye(4, dtype=dtype, device=device)
    world_H_base[:3, :3] = base_offset_rotation_matrix
    world_H_base[:3, 3] = base_offset_translation

    # Set the base transform on the chain
    chain.set_base_transform(world_H_base)
    print(f"Set robot base transform in world to:\n{world_H_base.cpu().numpy()}")

    # Target position in the WORLD coordinate system
    target_pos = torch.tensor([0.7, 0.2, 0.0], dtype=dtype, device=device)
    print(f"Target position (in world frame): {target_pos.cpu().numpy()}")

    # Initialize Joint Angles
    joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
    num_joints = len(joint_names)
    if num_joints == 0:
        print("Error: No non-fixed joints found in the chain. Cannot optimize.")
        return

    # Initialize with some non-zero values to avoid potential saddle points at zero
    # Ensure the number of initial angles matches num_joints
    initial_joint_angles = [0.1] * num_joints
    th = torch.tensor(initial_joint_angles, dtype=dtype, device=device).unsqueeze(0) # Batch size of 1
    th.requires_grad_(True)
    print(f"Initial joint angles for {num_joints} joints ({joint_names}): {th.detach().cpu().numpy()}")


    # Optimization Loop
    learning_rate = 0.05
    num_iterations = 201 # To print iter 0 and 200

    optimizer = torch.optim.Adam([th], lr=learning_rate)

    print("\nStarting optimization...")
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Forward kinematics
        try:
            transforms = chain.forward_kinematics(th)
        except Exception as e:
            print(f"Error during forward kinematics at iteration {i}: {e}")
            print(f"Current joint angles: {th.detach().cpu().numpy()}")
            return # Stop if FK fails

        if end_effector_name not in transforms:
            print(f"Error: End effector '{end_effector_name}' not found in transforms dictionary.")
            print(f"Available transform keys: {list(transforms.keys())}")
            return

        ee_matrix = transforms[end_effector_name].get_matrix()
        current_pos = ee_matrix[:, :3, 3]

        loss = torch.mean((current_pos - target_pos.unsqueeze(0))**2)

        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"Iteration {i:03d} | Loss: {loss.item():.6f} | "
                  f"Current Pos (World): {current_pos.squeeze().detach().cpu().numpy()} | "
                  f"Joints: {th.squeeze().detach().cpu().numpy()}")
            if th.grad is not None:
                print(f"    Grad Norm: {torch.linalg.norm(th.grad).item():.6f}")
            else:
                print(f"    Grad is None") # Should not happen if loss.backward() is called

    print("\nOptimization finished.")

    # Final Output
    final_transforms = chain.forward_kinematics(th)
    final_ee_matrix = final_transforms[end_effector_name].get_matrix()
    final_pos = final_ee_matrix[:, :3, 3]

    print(f"Initial joint angles: {initial_joint_angles}")
    print(f"Optimized joint angles: {th.detach().cpu().numpy().squeeze()}")
    print(f"Target position (in world frame): {target_pos.cpu().numpy()}")
    print(f"Final end-effector position (World): {final_pos.detach().cpu().numpy().squeeze()}")
    print(f"Final Loss: {loss.item():.6f}")
    print("\nrun_fk_optimization.py completed successfully.")

if __name__ == "__main__":
    run_fk_opt()
