#!/usr/bin/env python3
"""
MimicChain可视化示例
演示如何使用Open3D可视化五指机械手
"""
import torch
import numpy as np

import pytorch_kinematics as pk
from pytorch_kinematics import MimicChain
from pytorch_kinematics import RobotVisualizer, visualize_robot_simple, create_joint_trajectory_animation


def demo_hand_visualization():
    """演示机械手可视化"""
    print("=" * 50)
    print("MimicChain + Open3D 可视化演示")
    print("=" * 50)
    
    try:
        # 加载机械手模型
        mimic_chain = pk.build_mimic_chain_from_urdf(open("hand/schunk_svh_hand_right.urdf", mode="rb").read())
        mimic_chain.print_mimic_info()
        
        # 创建可视化器
        visualizer = RobotVisualizer(mimic_chain, mesh_path_prefix="hand/")
        
        print("\n演示1: 静态可视化")
        print("-" * 30)
        
        # 获取独立关节数量
        n_independent = len(mimic_chain.get_independent_joint_names())
        print(f"独立关节数量: {n_independent}")
        print(f"独立关节: {mimic_chain.get_independent_joint_names()}")
        
        # 创建一个初始姿态（所有关节为0）
        joint_values_zero = torch.zeros(n_independent)
        print(f"零姿态关节值: {joint_values_zero}")
        
        # 可视化零姿态
        visualizer.visualize(
            joint_values_zero,
            window_name="机械手 - 零姿态",
            show_coordinate_frames=True,
            show_joint_spheres=True
        )
        
        print("\n演示2: 随机姿态可视化")
        print("-" * 30)
        
        # 创建一个随机姿态
        joint_values_random = torch.rand(n_independent) * 0.8 - 0.4  # -0.4 到 0.4 弧度
        print(f"随机姿态关节值: {joint_values_random}")
        
        # 可视化随机姿态
        visualizer.visualize(
            joint_values_random,
            window_name="机械手 - 随机姿态",
            show_coordinate_frames=True,
            show_joint_spheres=True
        )
        
        print("\n演示3: 抓取姿态可视化")
        print("-" * 30)
        
        # 创建一个模拟抓取的姿态
        grasp_values = torch.zeros(n_independent)
        # 假设前几个关节控制手指弯曲
        if n_independent >= 5:
            grasp_values[:5] = torch.tensor([0.8, 0.8, 0.8, 0.8, 0.8])  # 手指弯曲
        
        print(f"抓取姿态关节值: {grasp_values}")
        
        # 可视化抓取姿态
        visualizer.visualize(
            grasp_values,
            window_name="机械手 - 抓取姿态",
            show_coordinate_frames=True,
            show_joint_spheres=True
        )
        
        return mimic_chain, visualizer
        
    except FileNotFoundError:
        print("❌ 未找到URDF文件: hand/schunk_svh_hand_right.urdf")
        print("请确保文件存在且路径正确")
        return None, None
    except Exception as e:
        print(f"❌ 可视化演示失败: {e}")
        return None, None


def demo_hand_animation():
    """演示机械手动画"""
    print("\n" + "=" * 50)
    print("机械手动画演示")
    print("=" * 50)
    
    try:
        # 加载机械手模型
        mimic_chain = pk.build_mimic_chain_from_urdf(open("hand/schunk_svh_hand_right.urdf", mode="rb").read())
        
        # 获取独立关节数量
        n_independent = len(mimic_chain.get_independent_joint_names())
        
        print(f"生成动画轨迹，独立关节数量: {n_independent}")
        
        # 生成一个简单的轨迹：从张开到闭合
        n_frames = 50
        trajectories = []
        
        for i in range(n_frames):
            # 线性插值从0到最大弯曲
            t = i / (n_frames - 1)  # 0 到 1
            
            joint_values = torch.zeros(n_independent)
            
            # 模拟手指逐渐弯曲的动作
            if n_independent >= 5:
                max_bend = 1.2  # 最大弯曲角度
                joint_values[:5] = t * max_bend
            
            trajectories.append(joint_values)
        
        print(f"生成了 {len(trajectories)} 帧的动画")
        
        # 创建动画
        create_joint_trajectory_animation(
            mimic_chain,
            trajectories,
            mesh_path_prefix="hand/",
            window_name="机械手动画 - 抓取动作",
            fps=10
        )
        
    except Exception as e:
        print(f"❌ 动画演示失败: {e}")


def demo_batch_visualization():
    """演示批量可视化"""
    print("\n" + "=" * 50)
    print("批量可视化演示")
    print("=" * 50)
    
    try:
        # 加载机械手模型
        mimic_chain = pk.build_mimic_chain_from_urdf(open("hand/schunk_svh_hand_right.urdf", mode="rb").read())
        
        # 获取独立关节数量
        n_independent = len(mimic_chain.get_independent_joint_names())
        
        # 创建批量关节值
        batch_size = 5
        batch_joint_values = torch.rand(batch_size, n_independent) * 1.0 - 0.5
        
        print(f"批量关节值形状: {batch_joint_values.shape}")
        
        # 计算批量正向运动学
        print("计算批量正向运动学...")
        batch_transforms = mimic_chain.forward_kinematics(batch_joint_values)
        
        print(f"✓ 成功计算了批量变换，坐标系数量: {len(batch_transforms)}")
        
        # 可视化第一个配置
        visualize_robot_simple(
            mimic_chain,
            batch_joint_values[0],
            mesh_path_prefix="hand/",
            window_name="批量可视化 - 第1个配置"
        )
        
    except Exception as e:
        print(f"❌ 批量可视化演示失败: {e}")


def demo_save_scene():
    """演示保存场景"""
    print("\n" + "=" * 50)
    print("保存场景演示")
    print("=" * 50)
    
    try:
        # 加载机械手模型
        mimic_chain = pk.build_mimic_chain_from_urdf(open("hand/schunk_svh_hand_right.urdf", mode="rb").read())
        
        # 创建可视化器
        visualizer = RobotVisualizer(mimic_chain, mesh_path_prefix="hand/")
        
        # 获取独立关节数量
        n_independent = len(mimic_chain.get_independent_joint_names())
        
        # 创建一个有趣的姿态
        joint_values = torch.zeros(n_independent)
        if n_independent >= 3:
            joint_values[:3] = torch.tensor([0.5, -0.3, 0.8])
        
        # 保存场景
        output_file = "hand_pose"
        visualizer.save_scene(joint_values, output_file, "ply")
        
        print(f"✓ 场景已保存到: {output_file}.ply")
        print("可以用MeshLab、Blender等软件打开查看")
        
    except Exception as e:
        print(f"❌ 保存场景失败: {e}")


def main():
    """主函数"""
    print("🤖 MimicChain Open3D 可视化演示")
    print("=" * 60)
    
    # 检查依赖
    try:
        import open3d as o3d
        print(f"✓ Open3D版本: {o3d.__version__}")
    except ImportError:
        print("❌ 请安装Open3D: pip install open3d")
        return
    
    # 演示1: 基本可视化
    chain, visualizer = demo_hand_visualization()
    
    if chain is None:
        print("由于URDF文件问题，跳过其他演示")
        return
    
    # 演示2: 动画可视化
    user_input = input("\n是否继续动画演示? (y/n): ").lower()
    if user_input == 'y':
        demo_hand_animation()
    
    # 演示3: 批量可视化
    user_input = input("\n是否继续批量可视化演示? (y/n): ").lower()
    if user_input == 'y':
        demo_batch_visualization()
    
    # 演示4: 保存场景
    user_input = input("\n是否保存场景到文件? (y/n): ").lower()
    if user_input == 'y':
        demo_save_scene()
    
    print("\n🎉 演示完成！")
    print("\n使用说明:")
    print("1. 导入: from pytorch_kinematics import visualize_robot_simple")
    print("2. 调用: visualize_robot_simple(chain, joint_values, mesh_path_prefix='hand/')")
    print("3. 对于MimicChain，记住只需要提供独立关节的值")


if __name__ == "__main__":
    main()
