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
        # Pass transform matrices directly instead of Transform3D objects
        obj_to_link_matrices = self.obj_frame_to_link_frame.get_matrix()

        # link_frame_to_obj_frame is a list of Transform3D objects
        # We need a way to pass their matrices. Stacking them might be an option if they are all (B,4,4) or (1,4,4)
        # Or pass them as a list of tensors. For autograd.Function, direct tensor inputs are better.
        # If link_frame_to_obj_frame[i] can have varying batch sizes (B_i, 4,4), stacking is hard.
        # Assuming they are compatible for stacking (e.g. all Bx4x4 or all 1x4x4 that can be expanded)
        # For simplicity, let's assume link_frame_to_obj_frame was constructed to have consistent batching
        # such that their matrices can be processed.
        # The original code indexed link_frame_to_obj_frame[i] and used it.
        # Let's pass the list of matrices. Autograd might not like list of tensors as direct input to save.
        # It's better to stack them if possible.
        # self.link_frame_to_obj_frame is a list of Transform3d, each could be batched.
        # Let's assume for now they are consistently batched as per self.tsf_batch

        # If self.link_frame_to_obj_frame is None or empty, handle it.
        # It's initialized in ComposedSDF.set_transforms based on obj_frame_to_link_frame.
        # m = tsf.get_matrix().inverse()
        # for i in range(S): self.link_frame_to_obj_frame.append(pk.Transform3d(matrix=m[self.ith_transform_slice(i)]))
        # So, each element is a Transform3D, whose matrix is a slice of m.
        # The batching of each element of link_frame_to_obj_frame corresponds to tsf_batch.
        # So we can stack their matrices.
        link_to_obj_matrices_list = [tf_inv.get_matrix() for tf_inv in self.link_frame_to_obj_frame]
        # Each matrix in list is (B,4,4) or (1,4,4 if B=1). Stack to (S,B,4,4) or (S,1,4,4)
        # Then reshape to (S*B, 4,4) if B > 1. Or (S,4,4) if B=1.
        # This needs to match how it's used in forward.

        # Let's simplify: pass the inverse of obj_to_link_matrices as link_to_obj_matrices
        # This assumes link_frame_to_obj_frame[i] is just the inverse of obj_frame_to_link_frame's i-th component transform.
        # This is true by construction in ComposedSDF.set_transforms.
        # obj_to_link_matrices is (S*B, 4, 4) or (S, 4, 4) if B=1.
        # Its inverse will have the same shape.

        # No, this is not quite right. link_frame_to_obj_frame is a list of S transforms,
        # each potentially batched (B,4,4).
        # Let's pass the list of matrices for now and handle it in forward.
        # This is not ideal for autograd.Function. Better to stack.

        # Stack matrices from link_frame_to_obj_frame
        # Each element of self.link_frame_to_obj_frame is a Transform3d object.
        # Its matrix is (tsf_batch_size, 4, 4). Let tsf_batch_size be B_tf.
        # So we'll have S tensors of shape (B_tf, 4, 4).
        # We can stack them into (S, B_tf, 4, 4).
        if self.link_frame_to_obj_frame:
            stacked_link_to_obj_matrices = torch.stack([tf.get_matrix() for tf in self.link_frame_to_obj_frame], dim=0)
        else: # Should not happen if sdfs is not empty
            stacked_link_to_obj_matrices = torch.empty(0, device=obj_to_link_matrices.device, dtype=obj_to_link_matrices.dtype)


        return ComposedSDFWithPartsFunction.apply(
            points_in_object_frame,
            self.sdfs, # sdfs list
            obj_to_link_matrices, # tensor
            self.tsf_batch, # tuple or None
            stacked_link_to_obj_matrices # tensor
        )


class ComposedSDFWithPartsFunction(Function):
    """
    Modified ComposedSDF function that returns:
    - pred: final SDF values (like original ComposedSDF)
    - pred_p_full: part-wise predictions for all links
    - closest_link_indices: which link is closest for each point
    """
    @staticmethod
    def forward(ctx, points_in_object_frame, sdfs,
                obj_to_link_matrices_arg, # Now a tensor
                tsf_batch,
                stacked_link_to_obj_matrices_arg # Now a tensor
                ):
        print_grad_info = True # Define at the top of the function
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
            # T_obj_to_link_flat_batch = obj_frame_to_link_frame # Old: was Transform3D object
            # Now obj_to_link_matrices_arg is the actual matrix tensor
            link_transform_matrices = obj_to_link_matrices_arg # Correctly use the input tensor.

            # points_in_object_frame (P_obj) has shape (B, N_pts, 3).
            # We need to prepare P_obj to be transformed by link_transform_matrices.
            # link_transform_matrices expects points with a leading batch dim of S*B or broadcastable.

            # Expand P_obj for S links: P_obj_expanded -> (S, B, N_pts, 3)
            P_obj_expanded_S_B_N_3 = points_in_object_frame.unsqueeze(0).expand(S, B, N_pts, 3)

            # Reshape to match the flat batch of transforms: P_obj_flat -> (S*B, N_pts, 3)
            P_obj_flat_SB_N_3 = P_obj_expanded_S_B_N_3.reshape(S * B, N_pts, 3)

            # Transform points manually to ensure gradient propagation from transforms
            # link_transform_matrices (batch S*B) transforms P_obj_flat_SB_N_3 (batch S*B, N_pts, 3)

            # link_transform_matrices = T_obj_to_link_flat_batch.get_matrix() # This line was the error, remove/corrected.

            # Extract rotation and translation components
            R = link_transform_matrices[:, :3, :3]  # Shape (S*B, 3, 3)
            t = link_transform_matrices[:, :3, 3]   # Shape (S*B, 3)

            if print_grad_info:
                print(f"link_transform_matrices: requires_grad={link_transform_matrices.requires_grad}, grad_fn={link_transform_matrices.grad_fn.name() if link_transform_matrices.grad_fn else 'None'}")
                print(f"R: requires_grad={R.requires_grad}, grad_fn={R.grad_fn.name() if R.grad_fn else 'None'}")
                print(f"t: requires_grad={t.requires_grad}, grad_fn={t.grad_fn.name() if t.grad_fn else 'None'}")

            # P_obj_flat_SB_N_3 has shape (S*B, N_pts, 3)
            # Manual transformation: R @ P_obj_flat_SB_N_3^T + t
            # Using einsum for batched matrix-vector multiply:
            # (S*B, N_pts, 3) = (S*B, 3, 3) @ (S*B, 3, N_pts) -> transpose P, then sum with t
            # P_transformed = torch.einsum('bij,bnj->bni', R, P_obj_flat_SB_N_3) + t.unsqueeze(1)
            # Or, more directly for P_obj_flat_SB_N_3 as (Batch, Num_points, 3_dim):
            # P_transformed_temp = (R @ P_obj_flat_SB_N_3.transpose(1, 2)).transpose(1, 2) # (S*B, N_pts, 3)
            # This is R @ P.T ; P is (..., N, D) so P.T is (..., D, N)
            # R is (..., D, D)
            # Result (..., D, N).transpose -> (..., N, D)

            # Alternative: Add homogeneous coordinate to points
            # P_homogeneous = torch.cat([P_obj_flat_SB_N_3, torch.ones(S*B, N_pts, 1, device=P_obj_flat_SB_N_3.device, dtype=P_obj_flat_SB_N_3.dtype)], dim=-1) # (S*B, N_pts, 4)
            # transformed_homogeneous = torch.einsum('bij,bnj->bni', link_transform_matrices, P_homogeneous) # (S*B, N_pts, 4)
            # pts_flat_SB_N_3 = transformed_homogeneous[:, :, :3] # (S*B, N_pts, 3)

            # Simpler: R*P + t, assuming P is (..., N, 3)
            # R is (..., 3, 3), t is (..., 3)
            # (R @ points.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)
            # P_obj_flat_SB_N_3 unsqueezed: (S*B, N_pts, 3, 1)
            # R: (S*B, 3, 3) -> needs to be (S*B, 1, 3, 3) for broadcasting with N_pts or expand R
            # R_expanded = R.unsqueeze(1).expand(-1, N_pts, -1, -1) # (S*B, N_pts, 3, 3)
            # rotated_points = (R_expanded @ P_obj_flat_SB_N_3.unsqueeze(-1)).squeeze(-1) # (S*B, N_pts, 3)
            # pts_flat_SB_N_3 = rotated_points + t.unsqueeze(1) # t unsqueezed to (S*B, 1, 3)

            # Using direct application of R and t, which is what Transform3D.transform_points does internally for non-homogeneous points
            # points are (batch, N, 3)
            # R is (batch, 3, 3)
            # t is (batch, 3)
            # Result = points @ R.transpose(-1, -2) + t.view(batch_size_of_transform, 1, 3)
            # This is for point transformation by p' = p @ R^T + t (if R is world_to_local)
            # Or p' = R @ p + t (if R is local_to_world) - this is typical for FK
            # transform_points in pytorch_kinematics.transforms.Transform3D:
            #   rot_points = self._matrix_rotation().matmul(points.unsqueeze(-1))
            #   return rot_points.squeeze(-1) + self._matrix_translation().unsqueeze(-2)
            # where _matrix_rotation is M[:, :3, :3] and _matrix_translation is M[:, :3, 3]

            pts_flat_SB_N_3 = R.bmm(P_obj_flat_SB_N_3.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)


            # Reshape pts_flat to the desired [S, B, N_pts, 3] structure
            pts = pts_flat_SB_N_3.view(S, B, N_pts, 3)
            # pts[s,b,n,:] is point n of batch b, transformed into the frame of link s (for batch b)

            # DEBUG GRADIENTS
            print_grad_info = True # Set to False to disable prints
            if print_grad_info:
                print("--- Debugging Gradients in ComposedSDFWithPartsFunction.forward ---")
                # obj_to_link_matrices_arg is the tensor directly
                if obj_to_link_matrices_arg.is_leaf:
                    print(f"obj_to_link_matrices_arg: leaf={obj_to_link_matrices_arg.is_leaf}, "
                          f"requires_grad={obj_to_link_matrices_arg.requires_grad}")
                else:
                    print(f"obj_to_link_matrices_arg: leaf={obj_to_link_matrices_arg.is_leaf}, "
                          f"requires_grad={obj_to_link_matrices_arg.requires_grad}, "
                          f"grad_fn={obj_to_link_matrices_arg.grad_fn.name() if obj_to_link_matrices_arg.grad_fn else 'None'}")

                print(f"P_obj_flat_SB_N_3: requires_grad={P_obj_flat_SB_N_3.requires_grad}, grad_fn={P_obj_flat_SB_N_3.grad_fn}")

                # R and t are derived from obj_to_link_matrices_arg
                # The manual transform uses obj_to_link_matrices_arg directly (via R and t)
                # So pts_flat_SB_N_3 is the result of these manual ops
                if pts_flat_SB_N_3.is_leaf:
                    print(f"pts_flat_SB_N_3 (output of manual transform): leaf={pts_flat_SB_N_3.is_leaf}, requires_grad={pts_flat_SB_N_3.requires_grad}")
                else:
                    print(f"pts_flat_SB_N_3 (output of manual transform): leaf={pts_flat_SB_N_3.is_leaf}, requires_grad={pts_flat_SB_N_3.requires_grad}, "
                          f"grad_fn={pts_flat_SB_N_3.grad_fn.name() if pts_flat_SB_N_3.grad_fn else 'None'}")

                if pts.is_leaf:
                     print(f"pts (reshaped): leaf={pts.is_leaf}, requires_grad={pts.requires_grad}")
                else:
                    print(f"pts (reshaped): leaf={pts.is_leaf}, requires_grad={pts.requires_grad}, "
                          f"grad_fn={pts.grad_fn.name() if pts.grad_fn else 'None'}")
                print("--------------------------------------------------------------------")


        # pts is now [S, B, N, 3] (if tsf_batch was (B,)) or [S, N, 3] (if tsf_batch was None and input pts was [N,3])
        
        sdfv_all_links = []  # List of S items, each [B, N] (or [N] if unbatched)
        sdfg_all_links = []  # List of S items, each [B, N, 3] (or [N,3] if unbatched)

        for i, sdf_i in enumerate(sdfs):
            # points_for_sdf_i: [B, N, 3] (or [N,3] if unbatched)
            # These are points transformed into the frame of sdf_i, for all batches.
            points_for_sdf_i = pts[i]
            
            if print_grad_info and i == 0: # Print for the first SDF only to avoid too much spam
                if points_for_sdf_i.is_leaf:
                    print(f"points_for_sdf_i (SDF input, i=0): leaf={points_for_sdf_i.is_leaf}, requires_grad={points_for_sdf_i.requires_grad}")
                else:
                    print(f"points_for_sdf_i (SDF input, i=0): leaf={points_for_sdf_i.is_leaf}, requires_grad={points_for_sdf_i.requires_grad}, "
                          f"grad_fn={points_for_sdf_i.grad_fn.name() if points_for_sdf_i.grad_fn else 'None'}")

            # v_i, g_i are computed by the individual sdf object (e.g., MeshSDF)
            # It should handle the batch dim of points_for_sdf_i ([B,N,3] -> [B,N], [B,N,3])
            v_i, g_i = sdf_i(points_for_sdf_i)

            if print_grad_info and i == 0:
                if v_i.is_leaf:
                    print(f"v_i (SDF output value, i=0): leaf={v_i.is_leaf}, requires_grad={v_i.requires_grad}")
                else:
                    print(f"v_i (SDF output value, i=0): leaf={v_i.is_leaf}, requires_grad={v_i.requires_grad}, "
                          f"grad_fn={v_i.grad_fn.name() if v_i.grad_fn else 'None'}")
            
            # link_frame_to_obj_frame[i] is a Transform3d.
            # If tsf_batch was (B,), its matrix is [B, 4, 4].
            # If tsf_batch was None, its matrix is [1, 4, 4] or [4,4].
            # g_i is [B, N, 3]. transform_normals should handle this:
            # - If transform is [B,3,3] and normals [B,N,3], it's batch-wise.
            # - If transform is [1,3,3] and normals [B,N,3], transform is broadcast.

            # stacked_link_to_obj_matrices_arg is (S, B_tf, 4, 4) or similar
            # We need the i-th inverse transform matrix: matrix_inv_i of shape (B_tf, 4, 4)
            matrix_inv_i = stacked_link_to_obj_matrices_arg[i]
            # Extract rotation part for transforming normals
            R_inv_i = matrix_inv_i[:, :3, :3] # (B_tf, 3, 3)
            # g_i is (B_tf, N_pts, 3)
            # Normal transformation: R_inv_i @ g_i (as column vectors)
            # g_i_in_obj_frame = pk.Transform3d(matrix=matrix_inv_i).transform_normals(g_i) # Old way
            g_i_in_obj_frame = R_inv_i.bmm(g_i.transpose(1,2)).transpose(1,2)


            if print_grad_info and i == 0:
                if g_i_in_obj_frame.is_leaf:
                    print(f"g_i_in_obj_frame (SDF output grad transformed, i=0): leaf={g_i_in_obj_frame.is_leaf}, requires_grad={g_i_in_obj_frame.requires_grad}")
                else:
                    print(f"g_i_in_obj_frame (SDF output grad transformed, i=0): leaf={g_i_in_obj_frame.is_leaf}, requires_grad={g_i_in_obj_frame.requires_grad}, "
                          f"grad_fn={g_i_in_obj_frame.grad_fn.name() if g_i_in_obj_frame.grad_fn else 'None'}")
            
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
        # These are v_all_links and g_all_links effectively.
        # closest needs to be [B, N] (or [N])
        closest = closest_flat.reshape(*pts_shape[:-1])

        if print_grad_info:
            print("--- Post-aggregation Gradients ---")
            if v.is_leaf:
                 print(f"v (stacked sdfv_all_links): leaf={v.is_leaf}, requires_grad={v.requires_grad}")
            else:
                print(f"v (stacked sdfv_all_links): leaf={v.is_leaf}, requires_grad={v.requires_grad}, "
                      f"grad_fn={v.grad_fn.name() if v.grad_fn else 'None'}")

            if vv.is_leaf:
                 print(f"vv (final SDF values): leaf={vv.is_leaf}, requires_grad={vv.requires_grad}")
            else:
                print(f"vv (final SDF values): leaf={vv.is_leaf}, requires_grad={vv.requires_grad}, "
                      f"grad_fn={vv.grad_fn.name() if vv.grad_fn else 'None'}")

            if g.is_leaf:
                print(f"g (stacked sdfg_all_links): leaf={g.is_leaf}, requires_grad={g.requires_grad}")
            else:
                print(f"g (stacked sdfg_all_links): leaf={g.is_leaf}, requires_grad={g.requires_grad}, "
                      f"grad_fn={g.grad_fn.name() if g.grad_fn else 'None'}")

            if gg.is_leaf:
                print(f"gg (final SDF gradients): leaf={gg.is_leaf}, requires_grad={gg.requires_grad}")
            else:
                print(f"gg (final SDF gradients): leaf={gg.is_leaf}, requires_grad={gg.requires_grad}, "
                      f"grad_fn={gg.grad_fn.name() if gg.grad_fn else 'None'}")
            print("--- End of ComposedSDFWithPartsFunction.forward Debug ---")

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
