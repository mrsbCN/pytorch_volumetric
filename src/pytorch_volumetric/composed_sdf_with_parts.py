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
            B = tsf_batch[0] # Batch size for transforms and points
            N_pts = pts_shape[-2] # Number of points per batch item, e.g., 500

            B = tsf_batch[0] # Batch size for transforms and points
            N_pts = pts_shape[-2] # Number of points per batch item

            # obj_frame_to_link_frame (T) has a matrix M of shape (S*B, 4, 4).
            # So, T itself has batch_shape (S*B).
            T_obj_to_link_flat_batch = obj_frame_to_link_frame

            # points_in_object_frame (P_obj) has shape (B, N_pts, 3).
            # We need to prepare P_obj to be transformed by T_obj_to_link_flat_batch.
            # T_obj_to_link_flat_batch expects points with a leading batch dim of S*B or broadcastable.

            # Expand P_obj for S links: P_obj_expanded -> (S, B, N_pts, 3)
            P_obj_expanded_S_B_N_3 = points_in_object_frame.unsqueeze(0).expand(S, B, N_pts, 3)

            # Reshape to match the flat batch of transforms: P_obj_flat -> (S*B, N_pts, 3)
            P_obj_flat_SB_N_3 = P_obj_expanded_S_B_N_3.reshape(S * B, N_pts, 3)

            # Transform points:
            # T_obj_to_link_flat_batch (batch S*B) transforms P_obj_flat_SB_N_3 (batch S*B, N_pts, 3)
            # pts_flat will have shape (S*B, N_pts, 3)
            pts_flat_SB_N_3 = T_obj_to_link_flat_batch.transform_points(P_obj_flat_SB_N_3)
            
            # Reshape pts_flat to the desired [S, B, N_pts, 3] structure
            pts = pts_flat_SB_N_3.view(S, B, N_pts, 3)
            # pts[s,b,n,:] is point n of batch b, transformed into the frame of link s (for batch b)


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
    @staticmethod
    def backward(ctx, grad_vv, grad_gg, grad_v_all_links, grad_g_all_links, grad_closest_link_indices):
        """
        Backward pass for ComposedSDFWithPartsFunction.
        This is a placeholder implementation based on ObjectFrameSDF.backward.
        It likely only correctly computes gradients for points_in_object_frame (input 0),
        and will return None for transform-related inputs, which is the source of the error.
        A full backward implementation is needed for transform gradients.
        """
        # final_sdf_grads_wrt_points_obj are the gradients of the final SDF values (vv)
        # with respect to the points in the object frame that were fed into MeshSDF,
        # but after selection of the closest link and transformation back to object frame.
        # This is 'gg' saved from the forward pass.
        final_sdf_grads_obj, = ctx.saved_tensors # 'gg' from forward

        # Gradient for the first input: points_in_object_frame
        # This input usually does not require grad in these scenarios.
        grad_input_points = None
        if ctx.needs_input_grad[0]:
            if grad_vv is not None: # dLoss/dvv
                # dLoss/d(points_in_object_frame) = dLoss/dvv * dvv/d(points_in_object_frame)
                # where dvv/d(points_in_object_frame) is final_sdf_grads_obj
                grad_input_points = grad_vv.unsqueeze(-1) * final_sdf_grads_obj

        # Gradients for other inputs (sdfs, obj_frame_to_link_frame, tsf_batch, link_frame_to_obj_frame)
        # are not computed here and will be None. This is why the error occurs if transforms need grad.
        # inputs to forward:
        # 0: points_in_object_frame
        # 1: sdfs
        # 2: obj_frame_to_link_frame
        # 3: tsf_batch
        # 4: link_frame_to_obj_frame

        # To fix the "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn",
        # we need to return a gradient for input 2 (obj_frame_to_link_frame) and/or input 4 (link_frame_to_obj_frame)
        # if they require gradients. PyTorch's autograd would then continue back to joint_config.
        # Calculating these gradients (grad_obj_to_link_tf, grad_link_to_obj_tf) is complex.

        return grad_input_points, None, None, None, None # Grad for inputs 0, 1, 2, 3, 4
