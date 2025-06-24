"""
Enhanced HandSDF with ContactGen-style part mapping and contact prediction
"""
import typing
import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
from pytorch_volumetric import sdf
from pytorch_volumetric.model_to_sdf import RobotSDF
from pytorch_volumetric.composed_sdf_with_parts import ComposedSDFWithParts, ComposedSDFWithPartsFunction
import logging

logger = logging.getLogger(__file__)

    # Hand part mapping aligned with ContactGen
LINK_TO_HAND_LABELS = {
        "right_hand_z": 13,
        "right_hand_a": 13,
        "right_hand_b": 14,
        "right_hand_c": 15,
        "right_hand_e2": 0,
        "right_hand_l": 1,
        "right_hand_virtual_l": 1,
        "right_hand_k": 4,
        "right_hand_j": 10,
        "right_hand_virtual_j": 10,
        "right_hand_i": 7,
        "right_hand_virtual_i": 7,
        "right_hand_p": 2,
        "right_hand_o": 5,
        "right_hand_n": 11,
        "right_hand_m": 8,
        "right_hand_t": 3,
        "right_hand_s": 6,
        "right_hand_r": 12,
        "right_hand_q": 9,
        "right_hand_base_link": 0,
        "right_hand_e1": 0,
        "right_hand_virtual_k": 4,
        "thtip": 15,
        "fftip": 3,
        "mftip": 6,
        "rftip": 12,
        "lftip": 9,
        "base_link": 0 
    }

class ContactHandSDF(RobotSDF):
    """
    Enhanced HandSDF for ContactGen-style grasp generation
    Provides:
    1. Part-wise contact predictions (pred_p_full)
    2. Final contact prediction (pred) 
    3. Part map alignment for ContactGen optimization
    """
    def __init__(self, chain: pk.Chain, default_joint_config=None, path_prefix='',
                 link_sdf_cls: typing.Callable[[sdf.ObjectFactory], sdf.ObjectFrameSDF] = sdf.MeshSDF,
                 contact_threshold=0.005):
        """
        Initialize ContactHandSDF
        
        :param chain: Robot description
        :param default_joint_config: Default joint configuration
        :param path_prefix: Path prefix for mesh files
        :param link_sdf_cls: SDF factory for each link
        :param contact_threshold: Threshold for contact probability computation
        """
        self.chain = chain
        self.dtype = self.chain.dtype
        self.device = self.chain.device
        self.q = None
        self.object_to_link_frames: typing.Optional[pk.Transform3d] = None
        self.joint_names = self.chain.get_joint_parameter_names()
        self.frame_names = self.chain.get_frame_names(exclude_fixed=False)
        self.sdf: typing.Optional[ComposedSDFWithParts] = None
        self.sdf_to_link_name = []
        self.configuration_batch = None

        sdfs = []
        offsets = []
        # get the link meshes from the frames and create meshes
        for frame_name in self.frame_names:
            frame = self.chain.find_frame(frame_name)
            # TODO create SDF for non-mesh primitives
            # TODO consider the visual offset transform
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    logger.info(f"{frame.link.name} offset {link_vis.offset}")
                    link_obj = sdf.MeshObjectFactory(link_vis.geom_param[0],
                                                     scale=link_vis.geom_param[1],
                                                     path_prefix=path_prefix)
                    link_sdf = link_sdf_cls(link_obj)
                    self.sdf_to_link_name.append(frame.link.name)
                    sdfs.append(link_sdf)
                    offsets.append(link_vis.offset)
                else:
                    logger.warning(f"Cannot handle non-mesh link visual type {link_vis} for {frame.link.name}")

        self.offset_transforms = offsets[0].stack(*offsets[1:]).to(device=self.device, dtype=self.dtype)
        self.sdf = ComposedSDFWithParts(sdfs, self.object_to_link_frames)
        self.set_joint_configuration(default_joint_config)
        
        # Create part mapping
        self.part_mapping = self._create_part_mapping()
        self.contact_threshold = contact_threshold
        self.n_parts = 16  # Total number of hand parts
    
    def _create_part_mapping(self):
        """
        Create mapping from link names to part IDs and build efficient mapping structures
        """
        part_map = {}
        link_to_part_list = []  # 从link索引到part ID的映射列表
        
        for link_idx, link_name in enumerate(self.sdf_to_link_name):
            if link_name in LINK_TO_HAND_LABELS:
                part_id = LINK_TO_HAND_LABELS[link_name]
                part_map[link_name] = part_id
                link_to_part_list.append(part_id)
            else:
                logger.warning(f"Link '{link_name}' not found in LINK_TO_HAND_LABELS, assigning to part 0")
                part_map[link_name] = 0
                link_to_part_list.append(0)
        
        # 创建link到part的映射张量
        self.link_to_part_tensor = torch.tensor(link_to_part_list, dtype=torch.long)
        
        # 为高效聚合创建part到link的反向映射
        self.part_to_links = {}
        for link_idx, part_id in enumerate(link_to_part_list):
            if part_id not in self.part_to_links:
                self.part_to_links[part_id] = []
            self.part_to_links[part_id].append(link_idx)
        
        logger.info(f"Created part mapping for {len(part_map)} links: {part_map}")
        logger.info(f"Part to links mapping: {self.part_to_links}")
        
        return part_map
    
    def _update_transforms_with_grad(self, joint_config):
        """
        Update transforms while preserving gradient flow
        This method recomputes the transforms directly without storing intermediate state
        """
        M = len(self.joint_names)
        
        # Handle batch dimensions
        if len(joint_config.shape) > 1:
            configuration_batch = joint_config.shape[:-1]
            joint_config_flat = joint_config.reshape(-1, M)
        else:
            configuration_batch = None
            joint_config_flat = joint_config
        
        # Forward kinematics with gradient preservation
        tf = self.chain.forward_kinematics(joint_config_flat, end_only=False)
        tsfs = []
        for link_name in self.sdf_to_link_name:
            tsfs.append(tf[link_name].get_matrix())
        
        # Handle offset transforms with compatible batch dimensions
        offset_tsf = self.offset_transforms.inverse()
        if configuration_batch is not None:
            expand_dims = (None,) * len(configuration_batch)
            offset_tsf_mat = offset_tsf.get_matrix()[(slice(None),) + expand_dims]
            offset_tsf_mat = offset_tsf_mat.repeat(1, *configuration_batch, 1, 1)
            offset_tsf = pk.Transform3d(matrix=offset_tsf_mat.reshape(-1, 4, 4))

        tsfs = torch.cat(tsfs)
        object_to_link_frames = offset_tsf.compose(pk.Transform3d(matrix=tsfs).inverse())
        
        # Update SDF transforms with gradient preservation
        self.sdf.set_transforms(object_to_link_frames, batch_dim=configuration_batch)
    
    def forward(self, points_in_object_frame, joint_config=None):
        """
        ContactGen-style forward pass with optional joint configuration
        
        :param points_in_object_frame: [..., N, 3] query points
        :param joint_config: Optional joint configuration to use for this forward pass
        :return: tuple of (pred, pred_p_full) where:
                 - pred: [..., N] final contact predictions (SDF-based)
                 - pred_p_full: [..., N, P] part-wise contact predictions
        """
        # If joint config is provided, compute transforms directly (preserving gradients)
        if joint_config is not None:
            sdf_vals, sdf_grads, sdf_per_link, sdf_per_link_grads, closest_link_indices = self._forward_with_dynamic_config(
                points_in_object_frame, joint_config)
        else:
            # Use cached transforms (standard operation)
            sdf_vals, sdf_grads, sdf_per_link, sdf_per_link_grads, closest_link_indices = self.sdf.forward_with_parts(points_in_object_frame)
        
        # Convert SDF to contact probabilities
        pred = torch.sigmoid(-sdf_vals / self.contact_threshold)
        
        # Create part-wise predictions aligned with hand parts
        # Transpose sdf_per_link from (S, ..., N) to (..., N, S)
        sdf_per_link_transposed = sdf_per_link.permute(list(range(1, sdf_per_link.ndim)) + [0])
        
        pred_p_full = self._create_part_aligned_predictions(
            sdf_per_link_transposed, closest_link_indices, points_in_object_frame.shape
        )
        
        return pred, pred_p_full
    
    def _forward_with_dynamic_config(self, points_in_object_frame, joint_config):
        """
        Forward pass with dynamic joint configuration (preserves gradients)
        This bypasses set_transforms to maintain gradient flow
        """
        M = len(self.joint_names)
        
        # Handle batch dimensions
        if len(joint_config.shape) > 1:
            configuration_batch = joint_config.shape[:-1]
            joint_config_flat = joint_config.reshape(-1, M)
        else:
            configuration_batch = None
            joint_config_flat = joint_config
        
        # Forward kinematics with gradient preservation
        tf = self.chain.forward_kinematics(joint_config_flat, end_only=False)
        tsfs = []
        for link_name in self.sdf_to_link_name:
            tsfs.append(tf[link_name].get_matrix())
        
        # Handle offset transforms with compatible batch dimensions
        offset_tsf = self.offset_transforms.inverse()
        if configuration_batch is not None:
            expand_dims = (None,) * len(configuration_batch)
            offset_tsf_mat = offset_tsf.get_matrix()[(slice(None),) + expand_dims]
            offset_tsf_mat = offset_tsf_mat.repeat(1, *configuration_batch, 1, 1)
            offset_tsf = pk.Transform3d(matrix=offset_tsf_mat.reshape(-1, 4, 4))

        tsfs = torch.cat(tsfs)
        object_to_link_frames = offset_tsf.compose(pk.Transform3d(matrix=tsfs).inverse())
        
        # Call ComposedSDFWithPartsFunction directly with gradient-enabled transforms
        return ComposedSDFWithPartsFunction.apply(
            points_in_object_frame, self.sdf.sdfs, object_to_link_frames, 
            configuration_batch, object_to_link_frames.inverse()
        )
    
    def _create_part_aligned_predictions(self, sdf_per_link, closest_link_indices, points_shape):
        """
        Create part-aligned contact predictions using efficient aggregation (参考ComposedSDF的最小值聚合)
        
        :param sdf_per_link: [..., N, S] SDF values for each link
        :param closest_link_indices: [..., N] closest link indices  
        :param points_shape: original points shape
        :return: [..., N, P] part-wise contact predictions
        """
        
        # 初始化part预测，设为负无穷以便取最小值
        pred_p_full = torch.full((*points_shape[:-1], self.n_parts), 
                                float('-inf'), 
                                device=sdf_per_link.device, 
                                dtype=sdf_per_link.dtype)
        
        # 高效的部位聚合：对每个part，找到其所有link并取最小值
        for part_id, link_indices in self.part_to_links.items():
            if link_indices:  # 确保该part有对应的link
                # 提取该part所有link的接触概率 [..., N, num_links_in_part]
                part_link_probs = sdf_per_link[..., link_indices]
                
                # 类似ComposedSDF取最小SDF值
                min_sdf, _ = torch.min(part_link_probs, dim=-1)  # [..., N]
                
                # 赋值给对应的part
                pred_p_full[..., part_id] = min_sdf
        
        # 将没有对应link的part设为0（从负无穷改为0）
        pred_p_full[pred_p_full == float('-inf')] = 0.0
        
        return pred_p_full
    
    def compute_contact_loss(self, points_in_object_frame, obj_cmap, obj_partition, w_contact=1e-1):
        """
        Compute ContactGen-style contact loss with part alignment
        
        :param points_in_object_frame: [..., N, 3] query points
        :param obj_cmap: [..., N] ground truth contact map
        :param obj_partition: [..., N] ground truth part assignments
        :param w_contact: contact loss weight
        :return: contact loss
        """
        # Forward pass
        pred, pred_p_full = self.forward(points_in_object_frame)
        
        # Extract part-specific predictions (ContactGen line 46 equivalent)
        pred_p = torch.gather(pred_p_full, dim=-1, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        
        # Compute contact loss
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        
        return loss_contact, pred, pred_p_full
    
    def query_contact_map(self, points_in_object_frame):
        """
        Query complete contact map information
        
        :param points_in_object_frame: [..., N, 3] query points
        :return: contact map dictionary
        """
        # Get SDF information
        sdf_vals, sdf_grads, sdf_per_link, sdf_per_link_grads, closest_link_indices = self.sdf.forward_with_parts(points_in_object_frame)
        
        # Convert to contact probabilities
        contact_prob = torch.sigmoid(-sdf_vals / self.contact_threshold)
        
        # Map link indices to part IDs using efficient tensor indexing
        link_to_part_tensor = self.link_to_part_tensor.to(device=closest_link_indices.device)
        part_ids = link_to_part_tensor[closest_link_indices]
        
        # Compute contact directions
        direction_map = -sdf_grads  # Point towards surface
        direction_map = F.normalize(direction_map, dim=-1)
        
        # Compute closest points
        closest_points = points_in_object_frame - sdf_vals.unsqueeze(-1) * sdf_grads
        
        return {
            'contact_prob': contact_prob,
            'part_map': part_ids,
            'direction_map': direction_map,
            'sdf_vals': sdf_vals,
            'closest_points': closest_points,
            'closest_link_indices': closest_link_indices
        }
    
    def optimize_grasp(self, obj_verts, obj_cmap, obj_partition, 
                      initial_joint_config, iterations=1000, lr=1e-3,
                      w_contact=1e-1, w_penetration=3.0, eps=-1e-3):
        """
        ContactGen-style grasp optimization
        
        :param obj_verts: [B, N, 3] object vertices
        :param obj_cmap: [B, N] contact map
        :param obj_partition: [B, N] part assignments
        :param initial_joint_config: initial joint configuration
        :param iterations: optimization iterations
        :param lr: learning rate
        :param w_contact: contact loss weight
        :param w_penetration: penetration loss weight
        :param eps: penetration threshold
        :return: optimized joint configuration
        """
        # Setup optimization
        joint_config = initial_joint_config.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([joint_config], lr=lr)
        
        for it in range(iterations):
            optimizer.zero_grad()
            
            # Compute predictions with joint configuration (preserves gradient flow)
            pred, pred_p_full = self.forward(obj_verts, joint_config)
            
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
        
        return joint_config
    
    def get_part_names(self):
        """Get mapping from part IDs to readable names"""
        return self.part_mapping
