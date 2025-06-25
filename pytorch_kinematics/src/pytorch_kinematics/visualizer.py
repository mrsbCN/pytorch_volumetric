"""
基于pytorch_kinematics.Chain和Open3D的机器人可视化模块
"""

import numpy as np
import torch
import sys
import open3d as o3d
from typing import Dict, List, Optional, Union
import os
import copy
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Some mesh loading features may be limited.")


class RobotVisualizer:
    """
    基于Chain的正向运动学结果进行机器人可视化
    """
    
    def __init__(self, chain, urdf_path: Optional[str] = None, mesh_path_prefix: str = ""):
        """
        初始化机器人可视化器
        
        Args:
            chain: pytorch_kinematics Chain对象
            urdf_path: URDF文件路径（用于解析mesh路径）
            mesh_path_prefix: mesh文件的路径前缀
        """
        self.chain = chain
        self.urdf_path = urdf_path
        self.mesh_path_prefix = mesh_path_prefix
        
        # 可视化相关
        self.vis = None
        self.link_geometries = {}  # 存储每个link的几何体
        self.joint_spheres = {}    # 存储关节位置的球体
        self.coordinate_frames = {}  # 存储坐标系
        
        # 缓存
        self.cached_meshes = {}
        
        # 解析link的mesh信息
        self._parse_link_meshes()
    
    def _parse_link_meshes(self):
        """解析所有link的mesh信息"""
        self.link_mesh_info = {}
        
        for frame_name in self.chain.get_frame_names(exclude_fixed=False):
            frame = self.chain.find_frame(frame_name)
            if frame and frame.link and frame.link.visuals:
                mesh_files = []
                for visual in frame.link.visuals:
                    if visual.geom_type == "mesh":
                        mesh_file = visual.geom_param[0]  # mesh文件路径
                        scale = visual.geom_param[1] if len(visual.geom_param) > 1 else [1.0, 1.0, 1.0]
                        if scale is None:
                            scale = [1.0, 1.0, 1.0]
                        mesh_files.append({
                            'file': mesh_file,
                            'scale': scale,
                            'offset': visual.offset
                        })
                
                if mesh_files:
                    self.link_mesh_info[frame.link.name] = mesh_files
    
    def _load_mesh(self, mesh_file: str, scale: List[float] = [1.0, 1.0, 1.0]) -> Optional[o3d.geometry.TriangleMesh]:
        """
        加载mesh文件
        
        Args:
            mesh_file: mesh文件路径
            scale: 缩放比例
        
        Returns:
            Open3D三角mesh对象
        """
        # 缓存检查
        cache_key = (mesh_file, tuple(scale))
        if cache_key in self.cached_meshes:
            return copy.deepcopy(self.cached_meshes[cache_key])
        
        # 构建完整路径
        if os.path.isabs(mesh_file):
            full_path = mesh_file
        else:
            full_path = os.path.join(self.mesh_path_prefix, mesh_file)
        
        if not os.path.exists(full_path):
            print(f"Warning: Mesh file not found: {full_path}")
            return None
        
        try:
            # 尝试用Open3D直接加载
            mesh = o3d.io.read_triangle_mesh(full_path)
            
            # 如果Open3D失败，尝试用trimesh加载
            if len(mesh.vertices) == 0 and TRIMESH_AVAILABLE:
                trimesh_obj = trimesh.load(full_path)
                if hasattr(trimesh_obj, 'vertices'):
                    mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
            
            if len(mesh.vertices) == 0:
                print(f"Warning: Failed to load mesh: {full_path}")
                return None
            
            # 应用缩放
            if scale != [1.0, 1.0, 1.0]:
                mesh.scale(scale, center=mesh.get_center())
            
            # 计算法向量
            mesh.compute_vertex_normals()
            
            # 缓存
            self.cached_meshes[cache_key] = copy.deepcopy(mesh)
            
            return mesh
            
        except Exception as e:
            print(f"Error loading mesh {full_path}: {e}")
            return None
    
    def _create_coordinate_frame(self, size: float = 0.1) -> o3d.geometry.TriangleMesh:
        """创建坐标系几何体"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def _create_joint_sphere(self, radius: float = 0.02) -> o3d.geometry.TriangleMesh:
        """创建关节球体"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        return sphere
    
    def _create_simple_link_geometry(self, length: float = 0.001) -> o3d.geometry.TriangleMesh:
        """为没有mesh的link创建简单几何体"""
        # 创建一个简单的圆柱体代表link
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.001, height=length)
        cylinder.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
        return cylinder
    
    def create_scene_geometries(self, joint_values: Union[torch.Tensor, np.ndarray, dict, list]) -> List[o3d.geometry.Geometry3D]:
        """
        基于给定关节值创建场景几何体
        
        Args:
            joint_values: 关节值
        
        Returns:
            几何体列表
        """
        # 计算正向运动学
        transforms_dict = self.chain.forward_kinematics(joint_values, end_only=False)
        
        geometries = []
        
        # 为每个frame创建几何体
        for frame_name, transform in transforms_dict.items():
            frame = self.chain.find_frame(frame_name)
            if not frame:
                continue
            
            # 获取变换矩阵
            if hasattr(transform, 'get_matrix'):
                T = transform.get_matrix().squeeze().cpu().numpy()
            else:
                T = transform.squeeze().cpu().numpy()
            
            # 添加坐标系
            coord_frame = self._create_coordinate_frame(size=0.03)
            coord_frame.transform(T)
            geometries.append(coord_frame)
            
            # 添加关节球体（如果是非固定关节）
            if frame.joint.joint_type != 'fixed':
                joint_sphere = self._create_joint_sphere(radius=0.005)
                joint_sphere.transform(T)
                geometries.append(joint_sphere)
            
            # 添加link几何体
            link_name = frame.link.name
            if link_name in self.link_mesh_info:
                # 有mesh文件
                for mesh_info in self.link_mesh_info[link_name]:
                    mesh = self._load_mesh(mesh_info['file'], mesh_info['scale'])
                    if mesh is not None:
                        # 应用offset变换
                        if mesh_info['offset'] is not None:
                            offset_matrix = mesh_info['offset'].get_matrix().squeeze().cpu().numpy()
                            mesh.transform(offset_matrix)
                        
                        # 应用frame变换
                        mesh.transform(T)
                        
                        # 设置颜色
                        if not mesh.has_vertex_colors():
                            mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 淡蓝色
                        
                        geometries.append(mesh)
            else:
                # 没有mesh，创建简单几何体
                simple_geom = self._create_simple_link_geometry()
                simple_geom.transform(T)
                geometries.append(simple_geom)
        
        return geometries
    
    
    def create_scene_geometries_base_trans(self, base_trans_values: Union[torch.Tensor, np.ndarray, dict, list]) -> List[o3d.geometry.Geometry3D]:
        """
        基于给定关节值创建场景几何体
        
        Args:
            joint_values: 关节值
        
        Returns:
            几何体列表
        """
        self.chain.set_base_transform(base_trans_values)

        # 计算正向运动学
        n_independent = len(self.chain.get_independent_joint_names())
        joint_values_zero = torch.zeros(n_independent)
        transforms_dict = self.chain.forward_kinematics(joint_values_zero, end_only=False)
        
        geometries = []
        
        # 为每个frame创建几何体
        for frame_name, transform in transforms_dict.items():
            frame = self.chain.find_frame(frame_name)
            if not frame:
                continue
            
            # 获取变换矩阵
            if hasattr(transform, 'get_matrix'):
                T = transform.get_matrix().squeeze().cpu().numpy()
            else:
                T = transform.squeeze().cpu().numpy()
            
            # 添加坐标系
            coord_frame = self._create_coordinate_frame(size=0.03)
            coord_frame.transform(T)
            geometries.append(coord_frame)
            
            # 添加关节球体（如果是非固定关节）
            if frame.joint.joint_type != 'fixed':
                joint_sphere = self._create_joint_sphere(radius=0.005)
                joint_sphere.transform(T)
                geometries.append(joint_sphere)
            
            # 添加link几何体
            link_name = frame.link.name
            if link_name in self.link_mesh_info:
                # 有mesh文件
                for mesh_info in self.link_mesh_info[link_name]:
                    mesh = self._load_mesh(mesh_info['file'], mesh_info['scale'])
                    if mesh is not None:
                        # 应用offset变换
                        if mesh_info['offset'] is not None:
                            offset_matrix = mesh_info['offset'].get_matrix().squeeze().cpu().numpy()
                            mesh.transform(offset_matrix)
                        
                        # 应用frame变换
                        mesh.transform(T)
                        
                        # 设置颜色
                        if not mesh.has_vertex_colors():
                            mesh.paint_uniform_color([0.8, 0.8, 0.9])  # 淡蓝色
                        
                        geometries.append(mesh)
            else:
                # 没有mesh，创建简单几何体
                simple_geom = self._create_simple_link_geometry()
                simple_geom.transform(T)
                geometries.append(simple_geom)
        
        return geometries

    def visualize(self, joint_values: Union[torch.Tensor, np.ndarray, dict, list], 
                  window_name: str = "Robot Visualization",
                  show_coordinate_frames: bool = True,
                  show_joint_spheres: bool = True,
                  background_color: List[float] = [0.1, 0.1, 0.1]):
        """
        可视化机器人
        
        Args:
            joint_values: 关节值
            window_name: 窗口名称
            show_coordinate_frames: 是否显示坐标系
            show_joint_spheres: 是否显示关节球体
            background_color: 背景颜色
        """
        # 创建几何体
        geometries = self.create_scene_geometries(joint_values)
        
        # 过滤几何体
        filtered_geometries = []
        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                if geom.has_triangles():
                    # 检查是否是坐标系或关节球体
                    vertices = np.asarray(geom.vertices)
                    if len(vertices) < 10 and not show_coordinate_frames:
                        continue  # 跳过坐标系
                    if len(vertices) < 100 and geom.has_vertex_colors():
                        colors = np.asarray(geom.vertex_colors)
                        if np.allclose(colors[0], [1.0, 0.0, 0.0]) and not show_joint_spheres:
                            continue  # 跳过关节球体
                filtered_geometries.append(geom)
        
        # 创建可视化窗口
        o3d.visualization.draw_geometries(
            filtered_geometries,
            window_name=window_name,
            width=1024,
            height=768
        )
    
    def create_base_trans_animation(self, base_trans_trajectories: List[Union[torch.Tensor, np.ndarray, dict, list]],
                        window_name: str = "Robot Animation",
                        fps: int = 30) -> None:
        """
        创建动画可视化
        
        Args:
            joint_trajectories: 关节轨迹列表
            window_name: 窗口名称
            fps: 帧率
        """
        # 初始化可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768)
        
        # 添加初始几何体
        initial_geometries = self.create_scene_geometries_base_trans(base_trans_trajectories[0])
        geometry_refs = []
        
        for geom in initial_geometries:
            vis.add_geometry(geom)
            geometry_refs.append(geom)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # 动画循环
        for base_trans_values in base_trans_trajectories:
            # 更新几何体
            new_geometries = self.create_scene_geometries_base_trans(base_trans_values)
            
            # 更新每个几何体
            for i, (old_geom, new_geom) in enumerate(zip(geometry_refs, new_geometries)):
                if i < len(geometry_refs):
                    # 更新顶点和变换
                    old_geom.vertices = new_geom.vertices
                    old_geom.triangles = new_geom.triangles
                    if new_geom.has_vertex_normals():
                        old_geom.vertex_normals = new_geom.vertex_normals
                    if new_geom.has_vertex_colors():
                        old_geom.vertex_colors = new_geom.vertex_colors
                    
                    vis.update_geometry(old_geom)
            
            vis.poll_events()
            vis.update_renderer()
            
            # 控制帧率
            import time
            time.sleep(1.0 / fps)
        
        vis.destroy_window()

    def create_animation(self, joint_trajectories: List[Union[torch.Tensor, np.ndarray, dict, list]],
                        window_name: str = "Robot Animation",
                        fps: int = 30) -> None:
        """
        创建动画可视化
        
        Args:
            joint_trajectories: 关节轨迹列表
            window_name: 窗口名称
            fps: 帧率
        """
        # 初始化可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768)
        
        # 添加初始几何体
        initial_geometries = self.create_scene_geometries(joint_trajectories[0])
        geometry_refs = []
        
        for geom in initial_geometries:
            vis.add_geometry(geom)
            geometry_refs.append(geom)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # 动画循环
        for joint_values in joint_trajectories:
            # 更新几何体
            new_geometries = self.create_scene_geometries(joint_values)
            
            # 更新每个几何体
            for i, (old_geom, new_geom) in enumerate(zip(geometry_refs, new_geometries)):
                if i < len(geometry_refs):
                    # 更新顶点和变换
                    old_geom.vertices = new_geom.vertices
                    old_geom.triangles = new_geom.triangles
                    if new_geom.has_vertex_normals():
                        old_geom.vertex_normals = new_geom.vertex_normals
                    if new_geom.has_vertex_colors():
                        old_geom.vertex_colors = new_geom.vertex_colors
                    
                    vis.update_geometry(old_geom)
            
            vis.poll_events()
            vis.update_renderer()
            
            # 控制帧率
            import time
            time.sleep(1.0 / fps)
        
        vis.destroy_window()
    
    def save_scene(self, joint_values: Union[torch.Tensor, np.ndarray, dict, list],
                   filename: str, file_format: str = "ply") -> None:
        """
        保存场景到文件
        
        Args:
            joint_values: 关节值
            filename: 文件名
            file_format: 文件格式 ("ply", "obj", "stl")
        """
        geometries = self.create_scene_geometries(joint_values)
        
        # 合并所有mesh
        combined_mesh = o3d.geometry.TriangleMesh()
        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh) and geom.has_triangles():
                combined_mesh += geom
        
        # 保存
        if file_format.lower() == "ply":
            o3d.io.write_triangle_mesh(f"{filename}.ply", combined_mesh)
        elif file_format.lower() == "obj":
            o3d.io.write_triangle_mesh(f"{filename}.obj", combined_mesh)
        elif file_format.lower() == "stl":
            o3d.io.write_triangle_mesh(f"{filename}.stl", combined_mesh)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


def visualize_robot_simple(chain, joint_values, mesh_path_prefix: str = "", **kwargs):
    """
    简单的机器人可视化函数
    
    Args:
        chain: pytorch_kinematics Chain对象
        joint_values: 关节值
        mesh_path_prefix: mesh文件路径前缀
        **kwargs: 传递给visualize方法的其他参数
    """
    visualizer = RobotVisualizer(chain, mesh_path_prefix=mesh_path_prefix)
    visualizer.visualize(joint_values, **kwargs)

def create_base_trajectory_animation(chain, base_trajectories, mesh_path_prefix: str = "", **kwargs):
    """
    创建关节轨迹动画
    
    Args:
        chain: pytorch_kinematics Chain对象
        joint_trajectories: 机器人base坐标轨迹列表
        mesh_path_prefix: mesh文件路径前缀
        **kwargs: 传递给create_animation方法的其他参数
    """
    visualizer = RobotVisualizer(chain, mesh_path_prefix=mesh_path_prefix)
    visualizer.create_base_trans_animation(base_trajectories, **kwargs)


def create_joint_trajectory_animation(chain, joint_trajectories, mesh_path_prefix: str = "", **kwargs):
    """
    创建关节轨迹动画
    
    Args:
        chain: pytorch_kinematics Chain对象
        joint_trajectories: 关节轨迹列表
        mesh_path_prefix: mesh文件路径前缀
        **kwargs: 传递给create_animation方法的其他参数
    """
    visualizer = RobotVisualizer(chain, mesh_path_prefix=mesh_path_prefix)
    visualizer.create_animation(joint_trajectories, **kwargs)


# 使用示例
if __name__ == "__main__":
    import pytorch_kinematics as pk
    
    # 示例：可视化一个简单的机器人
    # 这里需要一个实际的URDF文件
    try:
        path = "hand/schunk_svh_hand_right.urdf"
        
        chain = pk.build_mimic_chain_from_urdf(open(path, mode="rb").read())
        
        # 创建随机关节值
        joint_names = chain.get_joint_parameter_names()
        joint_values = torch.rand(len(joint_names)) * 0.5
        
        # 可视化
        visualize_robot_simple(chain, joint_values, mesh_path_prefix="hand/",show_coordinate_frames=False, show_joint_spheres=False)
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        print("请确保有有效的URDF文件和mesh文件")
