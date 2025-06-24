import os
import torch
import pytorch_kinematics as pk

# Simple 2-link arm URDF string
urdf_string = """
<robot name="simple_arm">
    <link name="link0"> <!-- Base link -->
        <visual>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </visual>
    </link>
    <link name="link1">
        <visual>
            <geometry>
                <box size="1 0.1 0.1"/>
            </geometry>
        </visual>
    </link>
    <link name="link2">
        <visual>
            <geometry>
                <box size="0.5 0.1 0.1"/>
            </geometry>
        </visual>
    </link>
    <joint name="joint0" type="revolute"> <!-- Joint connecting base to link1 -->
        <parent link="link0"/>
        <child link="link1"/>
        <origin xyz="0 0 0"/>
        <axis xyz="0 0 1"/>
    </joint>
    <joint name="joint1" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="1 0 0"/> <!-- Origin of joint1 on link1 -->
        <axis xyz="0 0 1"/>
    </joint>
</robot>
"""

def run_gradient_test():
    print("Building chain from URDF string...")
    chain = pk.build_chain_from_urdf(urdf_string)

    # Move chain to device and dtype
    dtype = torch.float64
    device = "cpu" # can be "cuda" if available
    print(f"Moving chain to {device} with {dtype}...")
    chain = chain.to(dtype=dtype, device=device)

    # Get joint names
    joint_names = chain.get_joint_parameter_names()
    if not joint_names:
        print("Error: No joint parameter names found. Ensure your URDF has non-fixed joints.")
        return
    print(f"Joint names: {joint_names}")

    # Create joint angle tensor
    num_joints = len(joint_names)
    print(f"Number of joints: {num_joints}")
    th = torch.randn(1, num_joints, dtype=dtype, device=device, requires_grad=True)
    print(f"Initial joint angles (th): {th}")

    # Perform forward kinematics
    print("Performing forward kinematics...")
    try:
        transforms = chain.forward_kinematics(th)
    except Exception as e:
        print(f"Error during forward kinematics: {e}")
        return

    # Select a transform matrix
    # Ensure there are links returned by get_link_names()
    link_names = chain.get_link_names()
    if not link_names:
        print("Error: No link names found in the chain.")
        return

    # Attempt to get the transform for the last link by name.
    # The URDF defines link0, link1, link2. After parsing, names might be chain_link0, etc.
    # We need to find the correct name as known by the chain object.

    # Let's try to get the transform for "link2" if it exists, otherwise the last one.
    # The actual link names might be prefixed by the robot name, e.g., "simple_arm_link2"
    # or might depend on how the parser names them.
    # A robust way is to use the names returned by chain.get_link_names()

    target_link_name = None
    # Check for specific link names that should exist based on the URDF
    possible_target_names = [name for name in link_names if "link2" in name]
    if not possible_target_names: # Fallback to the last link if "link2" isn't found (it should be)
        print(f"Warning: Could not find a link named 'link2'. Using last link: {link_names[-1]}")
        target_link_name = link_names[-1]
    else:
        target_link_name = possible_target_names[0] # Use the first match for "link2"

    print(f"Selected link for transform: {target_link_name}")

    if target_link_name not in transforms:
        print(f"Error: Target link '{target_link_name}' not found in transforms dictionary.")
        print(f"Available transform keys: {transforms.keys()}")
        return

    matrix = transforms[target_link_name].get_matrix()
    print(f"Transform matrix for {target_link_name}:\n{matrix}")

    # Compute dummy scalar loss
    loss = matrix.sum()
    print(f"Computed loss: {loss.item()}")

    # Perform backpropagation
    print("Performing backpropagation...")
    loss.backward()

    # Check if gradients are computed for th
    assert th.grad is not None, "Gradients are None for th"
    print("Assertion passed: th.grad is not None.")

    # Check if gradients are non-zero
    assert torch.any(th.grad != 0), "Gradients are all zero for th. This might indicate a problem."
    print("Assertion passed: Some gradients in th.grad are non-zero.")

    print("\nGradient flow test successful!")
    print(f"th.grad: {th.grad}")

if __name__ == "__main__":
    run_gradient_test()
