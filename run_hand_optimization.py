import torch
import pytorch_kinematics as pk
import os
from pytorch_volumetric.contact_robot_hand_sdf import ContactHandSDF
# from pytorch_volumetric import sdf # Not strictly needed if ContactHandSDF handles MeshObjectFactory

# IMPORTANT SETUP INSTRUCTIONS:
# 1. Navigate to the `pytorch_kinematics` directory and run:
#    pip install -e .
# 2. Navigate to the root directory of this repository (where `pytorch_volumetric`'s setup file is) and run:
#    pip install -e .
# This ensures that the modified code is used.

def run_optimization():
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Dummy Mesh Files
    print("Creating dummy mesh files...")
    os.makedirs("dummy_meshes", exist_ok=True)
    cube_obj_data = """
v -0.05 -0.01 -0.01
v  0.05 -0.01 -0.01
v  0.05  0.01 -0.01
v -0.05  0.01 -0.01
v -0.05 -0.01  0.01
v  0.05 -0.01  0.01
v  0.05  0.01  0.01
v -0.05  0.01  0.01
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
    with open("dummy_meshes/link1.obj", "w") as f:
        f.write(cube_obj_data)
    with open("dummy_meshes/link2.obj", "w") as f:
        f.write(cube_obj_data) # Using same cube for simplicity
    with open("dummy_meshes/fingertip.obj", "w") as f:
        f.write(cube_obj_data) # Using same cube for simplicity
    print("Dummy mesh files created in dummy_meshes/")

    # Load Robot Hand Model (Simple URDF with Mesh Visuals)
    print("Loading robot hand model...")
    hand_urdf_string = """
<robot name="simple_finger">
    <link name="base_link"/>
    <link name="link1">
        <visual><geometry><mesh filename="link1.obj"/></geometry></visual>
        <collision><geometry><mesh filename="link1.obj"/></geometry></collision>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="link1"/>
        <origin xyz="0 0 0"/>
        <axis xyz="0 0 1"/>
            <limit lower="-1.57" upper="1.57" effort="1.0" velocity="1.0"/>
    </joint>
    <link name="link2">
        <visual><geometry><mesh filename="link2.obj"/></geometry></visual>
        <collision><geometry><mesh filename="link2.obj"/></geometry></collision>
    </link>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0.1 0 0"/>
        <axis xyz="0 0 1"/>
            <limit lower="-1.57" upper="1.57" effort="1.0" velocity="1.0"/>
    </joint>
    <link name="fingertip">
        <visual><geometry><mesh filename="fingertip.obj"/></geometry></visual>
        <collision><geometry><mesh filename="fingertip.obj"/></geometry></collision>
    </link>
    <joint name="joint_tip" type="fixed">
        <parent link="link2"/>
        <child link="fingertip"/>
        <origin xyz="0.08 0 0"/>
    </joint>
</robot>
"""
    chain = pk.build_chain_from_urdf(hand_urdf_string)
    chain = chain.to(dtype=torch.float32, device=device)
    print("Robot chain built and moved to device.")

    # Initialize ContactHandSDF
    print("Initializing ContactHandSDF...")
    # Note: ContactHandSDF will print warnings for links not in its default part map, which is fine for this test.
    hand_sdf = ContactHandSDF(chain, path_prefix="dummy_meshes/", contact_threshold=0.01)
    hand_sdf = hand_sdf.to(dtype=torch.float32, device=device)
    print("ContactHandSDF initialized.")

    # Create Sample Object Point Cloud
    object_pts = torch.rand(1, 200, 3, device=device) * 0.1 + torch.tensor([0.1, 0.0, 0.0], device=device) # 1 batch, 200 points
    print(f"Object points created with shape: {object_pts.shape}")

    # Initial Joint Configuration
    # Ensure we only get names for joints that can be actuated
    joint_names_for_sdf = chain.get_joint_parameter_names(exclude_fixed=True)
    num_joints = len(joint_names_for_sdf)
    if num_joints == 0:
        print("Error: No non-fixed joints found in the chain. Cannot optimize.")
        return

    joint_angles = torch.zeros(1, num_joints, device=device, requires_grad=True)
    print(f"Initial joint angles for {num_joints} joints ({joint_names_for_sdf}) created with shape: {joint_angles.shape}")

    # Check if chain has any non-fixed joints for optimization
    if not joint_names_for_sdf:
        print("The robot chain has no non-fixed joints to optimize. Skipping optimization loop.")
    else:
        # Optimization Loop
        lr = 0.01
        iterations = 101 # Iterate 101 times to get 10 prints including iter 0 and 100
        optimizer = torch.optim.Adam([joint_angles], lr=lr)
        print("Starting optimization loop...")
        for i in range(iterations):
            optimizer.zero_grad()
            # Use the public forward method of ContactHandSDF which handles dynamic joint configs
            # The joint_config here should correspond to the joints ContactHandSDF expects based on its internal chain.
            pred_contact_prob, _ = hand_sdf.forward(object_pts, joint_config=joint_angles)

            # Loss: try to make SDF values zero, so contact probability (sigmoid output) should be 0.5
            loss = torch.mean((pred_contact_prob - 0.5)**2)

            loss.backward()

            # Gradient clipping to prevent explosion
            # torch.nn.utils.clip_grad_norm_([joint_angles], max_norm=1.0)

            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
                if joint_angles.grad is not None:
                    print(f"  Gradient norm: {torch.linalg.norm(joint_angles.grad).item():.6f}")
                else:
                    print(f"  Gradient is None for iteration {i}") # Should not happen if loss.backward() is called
        print("Optimization finished.")
        print(f"Optimized joint angles: {joint_angles.detach().cpu().numpy()}")

    print("run_hand_optimization.py completed successfully.")

if __name__ == "__main__":
    run_optimization()
