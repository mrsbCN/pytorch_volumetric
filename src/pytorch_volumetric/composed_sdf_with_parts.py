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
        pts_shape = points_in_object_frame.shape  # e.g., [B, N, 3]
        S = len(sdfs)

        if tsf_batch is None:
            # This case implies no batching of transforms; obj_frame_to_link_frame is [S, 4, 4]
            # and points_in_object_frame is [N, 3] or [B, N, 3] -> broadcast points to all S transforms
            # For simplicity, we'll assume tsf_batch is always present if points_in_object_frame is batched.
            # If points_in_object_frame is [N,3] and tsf_batch is None:
            if len(pts_shape) == 2: # [N,3]
                # pts will be [S, N, 3]
                pts = obj_frame_to_link_frame.transform_points(points_in_object_frame)
            else: # [B, N, 3] but tsf_batch is None. This is ambiguous.
                  # Defaulting to original behavior for this specific unbatched transform case,
                  # though it might still be problematic if B > 1.
                  # The user's problem implies batched transforms (tsf_batch is not None).
                points_in_object_frame_flat = points_in_object_frame.view(-1, 3)
                pts = obj_frame_to_link_frame.transform_points(points_in_object_frame_flat)
                # pts would be [S, B*N, 3]. Reshape to [S, B, N, 3]
                pts = pts.reshape(S, pts_shape[0], pts_shape[1], 3)

        else: # tsf_batch is not None, e.g., (B,)
            B = tsf_batch[0]
            N_pts = pts_shape[-2] # Number of points per batch item

            # obj_frame_to_link_frame's matrix is [S*B, 4, 4]
            # Order: (L0B0, L0B1, ..., L0B(B-1), L1B0, ..., LS-1B(B-1))
            
            all_pts_in_link_frame_batches = []
            # For each item in the point batch
            for b_idx in range(B):
                # points_b: [N_pts, 3]
                points_b = points_in_object_frame[b_idx]
                
                # Select S link transforms for this specific batch item b_idx
                # These are at flattened indices: b_idx, B+b_idx, 2B+b_idx, ..., (S-1)B+b_idx
                transform_indices_for_batch_b = torch.arange(S, device=obj_frame_to_link_frame.device) * B + b_idx
                
                # link_transforms_b_matrix: [S, 4, 4]
                link_transforms_b_matrix = obj_frame_to_link_frame.get_matrix()[transform_indices_for_batch_b]
                current_link_transforms_b = pk.Transform3d(matrix=link_transforms_b_matrix,
                                                           dtype=link_transforms_b_matrix.dtype,
                                                           device=link_transforms_b_matrix.device)
                
                # pts_b_in_link_frames: [S, N_pts, 3]
                # (transform_points broadcasts [N_pts,3] points to S transforms in current_link_transforms_b)
                pts_b_in_link_frames = current_link_transforms_b.transform_points(points_b)
                all_pts_in_link_frame_batches.append(pts_b_in_link_frames)

            # Stack to get [B, S, N_pts, 3]
            pts_stacked_by_batch = torch.stack(all_pts_in_link_frame_batches, dim=0)
            # Permute to [S, B, N_pts, 3] for iterating through SDFs first
            pts = pts_stacked_by_batch.permute(1, 0, 2, 3).contiguous()


        # pts is now [S, B, N, 3] (if tsf_batch was (B,)) or [S, N, 3] (if tsf_batch was None and input pts was [N,3])
        
        sdfv_all_links = []  # List of S items, each [B, N] (or [N] if unbatched)
        sdfg_all_links = []  # List of S items, each [B, N, 3] (or [N,3] if unbatched)

        for i, sdf_i in enumerate(sdfs):
            # points_for_sdf_i: [B, N, 3] (or [N,3] if unbatched)
            # These are points transformed into the frame of sdf_i, for all batches.
            points_for_sdf_i = pts[i]
            
            # v_i, g_i are computed by the individual sdf object (e.g., MeshSDF)
            # It should handle the batch dim of points_for_sdf_i ([B,N,3] -> [B,N], [B,N,3])
            v_i, g_i = sdf_i(points_for_sdf_i)
            
            # link_frame_to_obj_frame[i] is a Transform3d.
            # If tsf_batch was (B,), its matrix is [B, 4, 4].
            # If tsf_batch was None, its matrix is [1, 4, 4] or [4,4].
            # g_i is [B, N, 3]. transform_normals should handle this:
            # - If transform is [B,3,3] and normals [B,N,3], it's batch-wise.
            # - If transform is [1,3,3] and normals [B,N,3], transform is broadcast.
            g_i_in_obj_frame = link_frame_to_obj_frame[i].transform_normals(g_i)
            
            sdfv_all_links.append(v_i)
            sdfg_all_links.append(g_i_in_obj_frame)

        # Stack results from all links:
        # v: [S, B, N] (or [S,N] if unbatched)
        v = torch.stack(sdfv_all_links, dim=0)
        # g: [S, B, N, 3] (or [S,N,3] if unbatched)
        g = torch.stack(sdfg_all_links, dim=0)

        # Reshape for finding the minimum SDF value across links (dim 0)
        # For example, if v is [S, B, N], v_reshaped becomes [S, B*N]
        v_reshaped = v.reshape(S, -1)
        # closest_flat: [B*N,] (indices along S dimension)
        closest_flat = torch.argmin(torch.abs(v_reshaped), dim=0)

        # Index into v_reshaped and g_reshaped (g needs to be [S, B*N, 3])
        all_indices_flat = torch.arange(0, v_reshaped.shape[1], device=v.device)
        
        # vv_flat: [B*N,] - final SDF values
        vv_flat = v_reshaped[closest_flat, all_indices_flat]
        # g_reshaped: [S, B*N, 3]
        g_reshaped = g.reshape(S, -1, 3)
        # gg_flat: [B*N, 3] - final SDF gradients
        gg_flat = g_reshaped[closest_flat, all_indices_flat]

        # Reshape final outputs to match original point batching structure (pts_shape[:-1])
        # vv: [B, N] (or [N] if unbatched)
        vv = vv_flat.reshape(*pts_shape[:-1])
        # gg: [B, N, 3] (or [N,3] if unbatched)
        gg = gg_flat.reshape(*pts_shape[:-1], 3)
        
        # v and g are already [S, B, N] (or [S,N]) and [S, B, N, 3] (or [S,N,3]) respectively.
        # closest needs to be [B, N] (or [N])
        closest = closest_flat.reshape(*pts_shape[:-1])

        ctx.save_for_backward(gg)
        ctx.num_inputs = 5  # points_in_object_frame, sdfs, obj_frame_to_link_frame, tsf_batch, link_frame_to_obj_frame
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
