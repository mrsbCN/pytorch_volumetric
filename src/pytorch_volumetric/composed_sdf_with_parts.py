"""
Extended ComposedSDF that returns both SDF values and part-wise predictions
"""
import torch
import pytorch_kinematics as pk
from pytorch_volumetric.sdf import ComposedSDF
from torch.autograd import Function
import typing


class ComposedSDFWithParts(ComposedSDF):
    """
    Extended ComposedSDF that returns both SDF values and part-wise contact predictions
    This enables ContactGen-style part-conditioned contact prediction
    """
    
    def __init__(self, sdfs, obj_frame_to_each_frame=None):
        super().__init__(sdfs, obj_frame_to_each_frame)
    
    def forward_with_parts(self, points_in_object_frame):
        """
        Forward pass that returns both SDF values and part-wise predictions
        
        :param points_in_object_frame: [..., N, 3] query points
        :return: tuple of (sdf_vals, sdf_grads, part_predictions, closest_link_indices)
        """
        return ComposedSDFWithPartsFunction.apply(
            points_in_object_frame, self.sdfs, self.obj_frame_to_link_frame, 
            self.tsf_batch, self.link_frame_to_obj_frame
        )


class ComposedSDFWithPartsFunction(Function):
    """
    Modified ComposedSDF function that returns:
    - pred: final SDF values (like original ComposedSDF)  
    - pred_p_full: part-wise predictions for all links
    - closest_link_indices: which link is closest for each point
    """
    @staticmethod
    def forward(ctx, points_in_object_frame, sdfs, obj_frame_to_link_frame, tsf_batch, link_frame_to_obj_frame):
        pts_shape = points_in_object_frame.shape  # 保存原始形状，例如 [B, N, 3]
        # flatten it for the transform
        points_in_object_frame_flat = points_in_object_frame.view(-1, 3)  # [B*N, 3]
        flat_shape = points_in_object_frame_flat.shape
        S = len(sdfs)
        
        # pts[i] are now points in the ith SDF's frame
        pts = obj_frame_to_link_frame.transform_points(points_in_object_frame_flat)
        
        # S x B*N x 3
        if tsf_batch is not None:
            pts = pts.reshape(S, *tsf_batch, *flat_shape)
        
        sdfv = []
        sdfg = []
        for i, sdf in enumerate(sdfs):
            # B*N for v and B*N x 3 for g (or with tsf_batch dimensions)
            v, g = sdf(pts[i])
            # need to transform the gradient back to the object frame
            g = link_frame_to_obj_frame[i].transform_normals(g)
            sdfv.append(v)
            sdfg.append(g)

        # Concatenate all link predictions
        sdfv = torch.cat(sdfv)  # S*(B*N) or S*tsf_batch*(B*N)
        sdfg = torch.cat(sdfg)  # S*(B*N) x 3 or S*tsf_batch*(B*N) x 3

        # Reshape for processing: S x (B*N)
        v = sdfv.reshape(S, -1)  # S x (B*N)
        g = sdfg.reshape(S, -1, 3)  # S x (B*N) x 3
        
        # Find closest link for each point
        closest = torch.argmin(torch.abs(v), dim=0)  # (B*N,)

        all = torch.arange(0, v.shape[1], device=v.device)
        # Get final SDF values and gradients
        vv = v[closest, all]  # (B*N,)
        gg = g[closest, all]  # (B*N, 3)

        # 恢复原始维度
        # vv和gg恢复到原始batch shape
        vv = vv.reshape(*pts_shape[:-1])  # [B, N] 或 [..., N] 
        gg = gg.reshape(*pts_shape[:-1], 3)  # [B, N, 3] 或 [..., N, 3]
        
        # v和g需要在第一维度保留S (link数量)，然后恢复batch维度
        v = v.reshape(S, *pts_shape[:-1])  # [S, B, N] 或 [S, ..., N]
        g = g.reshape(S, *pts_shape[:-1], 3)  # [S, B, N, 3] 或 [S, ..., N, 3]
        
        # closest恢复到batch shape
        closest = closest.reshape(*pts_shape[:-1])  # [B, N] 或 [..., N]

        ctx.save_for_backward(gg)
        ctx.num_inputs = 5
        return vv, gg, v, g, closest
    
    # @staticmethod
    # def backward(ctx, grad_sdf_vals, grad_sdf_grads, grad_v, grad_g, grad_closest):
    #     """
    #     Backward pass for ComposedSDFWithPartsFunction
    #     只对points_in_object_frame计算梯度，其他参数返回None
    #     """
    #     # 获取保存的梯度信息
    #     final_sdf_grads, = ctx.saved_tensors
        
    #     # 只处理final SDF values的梯度
    #     if grad_sdf_vals is not None:
    #         # grad_sdf_vals shape: [..., N]
    #         # final_sdf_grads shape: [..., N, 3]
    #         dsdf_vals_dpoints = grad_sdf_vals.unsqueeze(-1) * final_sdf_grads
    #     else:
    #         dsdf_vals_dpoints = None
            
    #     # 返回所有输入参数的梯度，只有第一个(points)有梯度
    #     outputs = [None for _ in range(ctx.num_inputs)]
    #     outputs[0] = dsdf_vals_dpoints  # points_in_object_frame的梯度
    #     return tuple(outputs)
