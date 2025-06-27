from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics import jacobian
from pytorch_kinematics.frame import Frame, Link, Joint
from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_44, axis_and_d_to_pris_matrix


def get_n_joints(th):
    """

    Args:
        th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched

    Returns: The number of joints in the input

    """
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[-1]
    elif isinstance(th, list) or isinstance(th, dict):
        return len(th)
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def get_batch_size(th):
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[0]
    elif isinstance(th, dict):
        elem_shape = get_dict_elem_shape(th)
        return elem_shape[0]
    elif isinstance(th, list):
        # Lists cannot be batched. We don't allow lists of lists.
        return 1
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    return th, N


def get_dict_elem_shape(th_dict):
    elem = th_dict[list(th_dict.keys())[0]]
    if isinstance(elem, np.ndarray):
        return elem.shape
    elif isinstance(elem, torch.Tensor):
        return elem.shape
    else:
        return ()


class Chain:
    """
    Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint.
    """

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        self.identity = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.base_transform = tf.Transform3d(device=self.device, dtype=self.dtype)
        self.B_base = 1

        low, high = self.get_joint_limits()
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

        # As we traverse the kinematic tree, each frame is assigned an index.
        # We use this index to build a flat representation of the tree.
        # parents_indices and joint_indices all use this indexing scheme.
        # The root frame will be index 0 and the first frame of the root frame's children will be index 1,
        # then the child of that frame will be index 2, etc. In other words, it's a depth-first ordering.
        self.parents_indices = []  # list of indices from 0 (root) to the given frame
        self.joint_indices = []
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([self.n_joints, 3], dtype=self.dtype, device=self.device)
        self.link_offsets = []
        self.joint_offsets = []
        self.joint_type_indices = []
        queue = []
        queue.insert(-1, (self._root, -1, 0))  # the root has no parent so we use -1.
        idx = 0
        self.frame_to_idx = {}
        self.idx_to_frame = {}
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            name_strip = root.name.strip("\n")
            self.frame_to_idx[name_strip] = idx
            self.idx_to_frame[idx] = name_strip
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])

            is_fixed = root.joint.joint_type == 'fixed'

            if root.link.offset is None:
                self.link_offsets.append(None)
            else:
                self.link_offsets.append(root.link.offset.get_matrix())

            if root.joint.offset is None:
                self.joint_offsets.append(None)
            else:
                self.joint_offsets.append(root.joint.offset.get_matrix())

            if is_fixed:
                self.joint_indices.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(root.joint.name)
                self.axes[jnt_idx] = root.joint.axis
                self.joint_indices.append(jnt_idx)

            # these are integers so that we can use them as indices into tensors
            # FIXME: how do we know the order of these types in C++?
            self.joint_type_indices.append(Joint.TYPES.index(root.joint.joint_type))

            for child in root.children:
                queue.append((child, idx, depth + 1))

            idx += 1
        self.joint_type_indices = torch.tensor(self.joint_type_indices)
        self.joint_indices = torch.tensor(self.joint_indices)
        # We need to use a dict because torch.compile doesn't list lists of tensors
        self.parents_indices = [torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices]

    def set_base_transform(self, transform_matrix: torch.Tensor):
        """
        Sets the base transform for the kinematic chain.
        The input can be a single (4,4) matrix or a batch of (B,4,4) matrices.
        """
        if not isinstance(transform_matrix, torch.Tensor):
            raise TypeError(f"transform_matrix must be a torch.Tensor. Got {type(transform_matrix)}")

        if transform_matrix.ndim == 2:
            if transform_matrix.shape != (4, 4):
                raise ValueError(f"Input transform_matrix has shape {transform_matrix.shape}, but expected (4, 4) for a single matrix.")
            self.B_base = 1
        elif transform_matrix.ndim == 3:
            if transform_matrix.shape[1:] != (4, 4):
                raise ValueError(f"Input transform_matrix has shape {transform_matrix.shape}, but expected (B, 4, 4) for batched matrices.")
            if transform_matrix.shape[0] == 0:
                raise ValueError("Batch size for transform_matrix cannot be zero.")
            self.B_base = transform_matrix.shape[0]
        else:
            raise ValueError(f"Input transform_matrix has an invalid number of dimensions: {transform_matrix.ndim}. Expected 2 or 3.")

        # Store as Transform3d object, ensuring it's on the same device/dtype as the chain
        self.base_transform = tf.Transform3d(matrix=transform_matrix, device=self.device, dtype=self.dtype)


    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        if hasattr(self, 'base_transform') and self.base_transform is not None: # Check if initialized
            self.base_transform = self.base_transform.to(device=self.device, dtype=self.dtype)
        self.parents_indices = [p.to(dtype=torch.long, device=self.device) for p in self.parents_indices]
        self.joint_type_indices = self.joint_type_indices.to(dtype=torch.long, device=self.device)
        self.joint_indices = self.joint_indices.to(dtype=torch.long, device=self.device)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.joint_offsets = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in
                              self.joint_offsets]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)

        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame: Frame) -> Optional[Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name) -> Optional[Frame]:
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame) -> Optional[Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    @staticmethod
    def _get_joints(frame, exclude_fixed=True):
        joints = []
        if exclude_fixed and frame.joint.joint_type != "fixed":
            joints.append(frame.joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child))
        return joints

    def get_joints(self, exclude_fixed=True):
        joints = self._get_joints(self._root, exclude_fixed=exclude_fixed)
        return joints

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for j in self.get_joints(exclude_fixed=exclude_fixed):
            if exclude_fixed and j.joint_type == 'fixed':
                continue
            names.append(j.name)
        return names

    @staticmethod
    def _find_joint_recursive(name, frame):
        for child in frame.children:
            if child.joint.name == name:
                return child.joint
            ret = Chain._find_joint_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name) -> Optional[Link]:
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    def find_joint(self, name):
        if self._root.joint.name == name:
            return self._root.joint
        return self._find_joint_recursive(name, self._root)

    @staticmethod
    def _get_joint_parent_frame_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parent_frame_names(child, exclude_fixed))
        return joint_names

    def get_joint_parent_frame_names(self, exclude_fixed=True):
        names = self._get_joint_parent_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed=True) -> Sequence[str]:
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

    def get_frame_names(self, exclude_fixed=True):
        names = self._get_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_links(frame):
        links = [frame.link]
        for child in frame.children:
            links.extend(Chain._get_links(child))
        return links

    def get_links(self):
        links = self._get_links(self._root)
        return links

    @staticmethod
    def _get_link_names(frame):
        link_names = [frame.link.name]
        for child in frame.children:
            link_names.extend(Chain._get_link_names(child))
        return link_names

    def get_link_names(self):
        names = self._get_link_names(self._root)
        return sorted(set(names), key=names.index)

    @lru_cache
    def get_frame_indices(self, *frame_names):
        return torch.tensor([self.frame_to_idx[n] for n in frame_names], dtype=torch.long, device=self.device)

    def print_tree(self, do_print=True):
        tree = str(self._root)
        if do_print:
            print(tree)
        return tree

    def forward_kinematics(self, th, frame_indices: Optional = None):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            frame_indices: A list of frame indices to compute transforms for. If None, all frames are computed.
                Use `get_frame_indices` to convert from frame names to frame indices.

        Returns:
            A dict of Transform3d objects for each frame.

        """
        if frame_indices is None:
            frame_indices = self.get_all_frame_indices()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        B_th = th.shape[0]
        axes_expanded = self.axes.unsqueeze(0).repeat(B_th, 1, 1) # This self.axes is (n_joints, 3) from __init__ for actuated joints
                                                                  # So axes_expanded is (B_th, n_joints, 3)
                                                                  # th is (B_th, n_joints)

        # compute all joint transforms at once first for actuated joints
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th) # (B_th, n_joints, 4, 4)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th) # (B_th, n_joints, 4, 4)

        _initial_world_accumulator_matrix = self.base_transform.get_matrix().clone()
        if _initial_world_accumulator_matrix.ndim == 2:
            _initial_world_accumulator_matrix = _initial_world_accumulator_matrix.unsqueeze(0)

        # Ensure consistent batching between base_transform and th
        # B_base from __init__ or set_base_transform, B_th from current th
        # Target batch size for FK operations
        if self.B_base == 1 and B_th > 1:
            _initial_world_accumulator_matrix = _initial_world_accumulator_matrix.repeat(B_th, 1, 1)
            effective_batch_size = B_th
        elif self.B_base > 1 and B_th == 1:
            # th needs to be expanded to match base_transform's batch
            # This case should be handled by ensuring th is appropriately batched if base is batched.
            # For now, assume if B_base > 1, B_th must also be > 1 and equal, or B_th=1 will be broadcast.
            # The user's new FK code does not explicitly expand `th` if B_th=1 and B_base > 1.
            # It expands _initial_world_accumulator_matrix if B_base=1 and B_th > 1.
            # What if B_base > 1 and B_th = 1? Original FK would make th (B_base, num_joints).
            # The user's `ensure_tensor` makes `th` (1, num_joints) if input is single.
            # The user's FK then uses B_th = 1. This means rev_jnt_transform is (1, num_joints, 4,4).
            # If _initial_world_accumulator_matrix is (B_base, 4,4), matmul might broadcast but could be tricky.
            # Safest is to ensure they match or one is 1 and expands.
            # The user's code: `if self.B_base == 1 and B_th > 1: ... repeat(_initial_world_accumulator_matrix)`
            # `if self.B_base > 1 and B_th > 1 and self.B_base != B_th: raise Error`
            # This implies if B_th = 1 and B_base > 1, no error, _initial_world_accumulator_matrix is (B_base,4,4)
            # and joint transforms are (1,num_joints,4,4). Matmul broadcasting (B,N,M) @ (1,M,P) -> (B,N,P) is fine.
            effective_batch_size = self.B_base
        elif self.B_base > 1 and B_th > 1 and self.B_base != B_th:
             raise ValueError(f"Batch size of base_transform ({self.B_base}) and th ({B_th}) must be equal.")
        else: # B_base == B_th (both >1), or B_base=1, B_th=1
            effective_batch_size = B_th # or self.B_base, they are compatible

        frame_transforms = {}

        # Ensure frame_indices is iterable (it's a tensor from get_all_frame_indices)
        for frame_idx_tensor in frame_indices:
            frame_idx_item = frame_idx_tensor.item()

            # Start with the correctly batched initial world accumulator
            frame_transform_matrix = _initial_world_accumulator_matrix.clone()
            # If _initial_world_accumulator_matrix is (1,4,4) and effective_batch_size > 1, repeat it.
            # This ensures frame_transform_matrix starts with the correct batch dim for this FK call.
            if frame_transform_matrix.shape[0] == 1 and effective_batch_size > 1:
                 frame_transform_matrix = frame_transform_matrix.repeat(effective_batch_size, 1, 1)


            # Iterate through the frames in the path from root to current_frame_idx
            # self.parents_indices[frame_idx_item] contains indices of frames in the path from root to frame_idx_item (inclusive)
            for path_component_idx_tensor in self.parents_indices[frame_idx_item]:
                path_component_idx_item = path_component_idx_tensor.item()

                # If this component is already computed and stored, use it.
                # This is a more robust memoization than the user's original sketch.
                # However, the user's code structure `frame_transform = _initial_world_accumulator_matrix.clone()`
                # at the start of *each* `for frame_idx in frame_indices:` loop, and then iterating
                # `for chain_idx in self.parents_indices[frame_idx.item()]` means it recomputes the path.
                # To match that:
                # The `frame_transform_matrix` here is accumulated along one path.
                # The user's `if chain_idx.item() in frame_transforms:` was problematic.
                # The current structure: outer loop for each target frame, inner loop builds its path.

                # Apply link offset for path_component_idx_item
                link_offset_i = self.link_offsets[path_component_idx_item] # This is already a tensor or None
                if link_offset_i is not None:
                    frame_transform_matrix = frame_transform_matrix @ link_offset_i

                # Apply joint offset for path_component_idx_item
                joint_offset_i = self.joint_offsets[path_component_idx_item] # This is already a tensor or None
                if joint_offset_i is not None:
                    frame_transform_matrix = frame_transform_matrix @ joint_offset_i

                # Apply joint transform for path_component_idx_item
                # self.joint_indices maps frame index (path_component_idx_item) to an index in the `th` tensor (if actuated) or -1 (if fixed)
                # self.joint_type_indices maps frame index to joint type (0:fixed, 1:revolute, 2:prismatic)

                actuated_joint_idx_for_th = self.joint_indices[path_component_idx_item] # This is the index into `th`
                joint_type = self.joint_type_indices[path_component_idx_item]

                if joint_type == 0: # Fixed joint
                    pass
                elif joint_type == 1: # Revolute/Continuous
                    # rev_jnt_transform is (B_th, n_joints, 4, 4)
                    # actuated_joint_idx_for_th is scalar index for th
                    joint_transform_i = rev_jnt_transform[:, actuated_joint_idx_for_th] # (B_th, 4, 4)
                    # frame_transform_matrix is (effective_batch_size, 4,4)
                    # joint_transform_i is (B_th, 4,4)
                    # Broadcasting rules apply if effective_batch_size != B_th (one must be 1)
                    frame_transform_matrix = frame_transform_matrix @ joint_transform_i
                elif joint_type == 2: # Prismatic
                    joint_transform_i = pris_jnt_transform[:, actuated_joint_idx_for_th] # (B_th, 4, 4)
                    frame_transform_matrix = frame_transform_matrix @ joint_transform_i

            # After iterating through the whole path for frame_idx_item, store its world transform
            frame_transforms[frame_idx_item] = frame_transform_matrix

        # Convert stored matrices to Transform3d objects
        frame_names_and_transform3ds = {self.idx_to_frame[idx_val]: tf.Transform3d(matrix=matrix_val)
                                        for idx_val, matrix_val in frame_transforms.items()}
        return frame_names_and_transform3ds

    def ensure_tensor(self, th):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names(). th must contain all joints in the entire chain.
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, dict):
            # convert dict to a flat, complete, tensor of all joints values. Missing joints are filled with zeros.
            th_dict = th
            elem_shape = get_dict_elem_shape(th_dict)

            # Use get_joint_parameter_names() which might be overridden by MimicChain
            # self.n_joints refers to the number of joints this Chain instance expects (e.g., independent joints for MimicChain)
            current_n_joints = self.n_joints

            # Create a base tensor. If elem_shape is not empty, it means input `th` was a batch of dicts.
            # This part of ensure_tensor in original chain.py is a bit complex for batched dicts.
            # For safety, let's assume elem_shape is empty for now if th is dict (meaning one dict, not batch of dicts).
            # The user's FK code calls `th = torch.atleast_2d(th)` after `ensure_tensor`.
            if elem_shape: # Batch of dicts - this case is complex and not fully handled by original ensure_tensor for dicts easily
                # This part needs careful thought if we expect batch of dicts as input to FK
                # For now, assume single dict if it's a dict.
                pass # Fall through to non-batched dict handling

            _th = torch.zeros([current_n_joints], device=self.device, dtype=self.dtype) # Default to zeros

            # Use the potentially overridden get_joint_parameter_names()
            joint_names_for_chain = self.get_joint_parameter_names()

            all_keys_present = True
            for i, joint_name in enumerate(joint_names_for_chain):
                if joint_name in th_dict:
                    val = th_dict[joint_name]
                    # Ensure val is a tensor scalar or compatible
                    if not torch.is_tensor(val): val = torch.tensor(val, device=self.device, dtype=self.dtype)
                    if val.numel() != 1: raise ValueError(f"Value for joint {joint_name} in dict must be scalar.")
                    _th[i] = val
                else:
                    # This behavior (error on missing keys) is from original Chain.ensure_tensor
                    # The user's provided FK code's ensure_tensor was different.
                    # For consistency with existing Chain methods, let's require all keys.
                    all_keys_present = False # Or fill with NaN and check as original did.
                    # Let's match original chain.py: fill with NaN and error if any NaN.
                    _th[i] = torch.nan


            if not all_keys_present or torch.any(torch.isnan(_th)):
                missing_joints = [joint_names_for_chain[i] for i, val_t in enumerate(_th) if torch.isnan(val_t)]
                msg = f"Missing values for the following joints: {missing_joints}"
                raise ValueError(msg)
            th = _th

        # Original ensure_tensor from chain.py also had a check for th.ndim and th.shape[1] vs self.n_joints
        # after this block. Let's ensure th is 2D.
        if not torch.is_tensor(th): # If it was list/numpy and not dict, it became tensor above.
             th = torch.as_tensor(th, device=self.device, dtype=self.dtype)

        # This part is after the dict conversion in the original chain.py
        # It assumes th is now a tensor representing values for self.n_joints
        # if th.ndim == 0: th = th.view(1,1) # Should not happen if th represents multiple joints
        # elif th.ndim == 1: th = th.view(1, -1) # Make it (1, num_joints)
        #
        # if th.shape[-1] != self.n_joints:
        #     raise ValueError(f"Joint tensor th has incorrect number of joints. Expected {self.n_joints}, got {th.shape[-1]}.")

        return th # ensure_tensor in user's FK code calls atleast_2d(th) after this.

    def get_all_frame_indices(self):
        frame_indices = self.get_frame_indices(*self.get_frame_names(exclude_fixed=False))
        return frame_indices

    def clamp(self, th):
        """

        Args:
            th: Joint configuration

        Returns: Always a tensor in the order of self.get_joint_parameter_names(), possibly batched.

        """
        th = self.ensure_tensor(th)
        return torch.clamp(th, self.low, self.high)

    def get_joint_limits(self):
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self):
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self):
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(self, param_name):
        low = []
        high = []
        for joint in self.get_joints():
            val = getattr(joint, param_name)
            if val is None:
                # NOTE: This changes the previous default behavior of returning
                # +/- np.pi for joint limits to be more natural for both
                # revolute and prismatic joints
                low.append(-np.inf)
                high.append(np.inf)
            else:
                low.append(val[0])
                high.append(val[1])
        return low, high

    @staticmethod
    def _get_joints_and_child_links(frame):
        joint = frame.joint

        me_and_my_children = [frame.link]
        for child in frame.children:
            recursive_child_links = yield from Chain._get_joints_and_child_links(child)
            me_and_my_children.extend(recursive_child_links)

        if joint is not None and joint.joint_type != 'fixed':
            yield joint, me_and_my_children

        return me_and_my_children

    def get_joints_and_child_links(self):
        yield from Chain._get_joints_and_child_links(self._root)


class SerialChain(Chain):
    """
    A serial Chain specialization with no branches and clearly defined end effector.
    Serial chains can be generated from subsets of a Chain.
    """

    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        root_frame = chain._root if root_frame_name == "" else chain.find_frame(root_frame_name)
        if root_frame is None:
            raise ValueError("Invalid root frame name %s." % root_frame_name)
        chain = Chain(root_frame, **kwargs)

        # make a copy of those frames that includes only the chain up to the end effector
        end_frame_idx = chain.get_frame_indices(end_frame_name)
        ancestors = chain.parents_indices[end_frame_idx]

        frames = []
        # first pass create copies of the ancestor nodes
        for idx in ancestors:
            this_frame_name = chain.idx_to_frame[idx.item()]
            this_frame = copy.deepcopy(chain.find_frame(this_frame_name))
            if idx == end_frame_idx:
                this_frame.children = []
            frames.append(this_frame)
        # second pass assign correct children (only the next one in the frame list)
        for i in range(len(ancestors) - 1):
            frames[i].children = [frames[i + 1]]

        self._serial_frames = frames
        super().__init__(frames[0], **kwargs)

    def jacobian(self, th, locations=None, **kwargs):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations, **kwargs)

    def forward_kinematics(self, th, end_only: bool = True):
        """ Like the base class, except `th` only needs to contain the joints in the SerialChain, not all joints. """
        frame_indices, th = self.convert_serial_inputs_to_chain_inputs(th, end_only)

        mat = super().forward_kinematics(th, frame_indices)

        if end_only:
            return mat[self._serial_frames[-1].name]
        else:
            return mat

    def convert_serial_inputs_to_chain_inputs(self, th, end_only: bool):
        # th = self.ensure_tensor(th)
        th_b = get_batch_size(th)
        th_n_joints = get_n_joints(th)
        if isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)

        if end_only:
            frame_indices = self.get_frame_indices(self._serial_frames[-1].name)
        else:
            # pass through default behavior for frame indices being None, which is currently
            # to return all frames.
            frame_indices = None
        if th_n_joints < self.n_joints:
            # if th is only a partial list of joints, assume it's a list of joints for only the serial chain.
            partial_th = th
            nonfixed_serial_frames = list(filter(lambda f: f.joint.joint_type != 'fixed', self._serial_frames))
            if th_n_joints != len(nonfixed_serial_frames):
                raise ValueError(f'Expected {len(nonfixed_serial_frames)} joint values, got {th_n_joints}.')
            th = torch.zeros([th_b, self.n_joints], device=self.device, dtype=self.dtype)
            for i, frame in enumerate(nonfixed_serial_frames):
                joint_name = frame.joint.name
                if isinstance(partial_th, dict):
                    partial_th_i = partial_th[joint_name]
                else:
                    partial_th_i = partial_th[..., i]
                k = self.frame_to_idx[frame.name]
                jnt_idx = self.joint_indices[k]
                if frame.joint.joint_type != 'fixed':
                    th[..., jnt_idx] = partial_th_i
        return frame_indices, th


class MimicChain(Chain):
    """
    扩展Chain类以支持mimic关节的五指机械手，自动从URDF中解析mimic关系
    """
    
    def __init__(self, root_frame, mimic_config=None, dtype=torch.float32, device="cpu"):
        super().__init__(root_frame, dtype, device)
        
        # mimic_config可以手动指定，或者从URDF中自动解析
        self.mimic_config = mimic_config or {}
        self._build_mimic_matrix()
    
    def to(self, dtype=None, device=None):
        """重写to方法，确保mimic相关矩阵也被转移"""
        # 调用父类的to方法
        super().to(dtype, device)
        
        # 转移mimic相关的张量
        if hasattr(self, 'mimic_matrix'):
            self.mimic_matrix = self.mimic_matrix.to(dtype=self.dtype, device=self.device)
        if hasattr(self, 'mimic_offset'):
            self.mimic_offset = self.mimic_offset.to(dtype=self.dtype, device=self.device)
        return self

    def _build_mimic_matrix(self):
        """构建mimic关节映射矩阵和offset向量"""
        all_joints = super().get_joint_parameter_names()
        n_joints = len(all_joints)
        
        # 区分独立关节和mimic关节
        self.independent_joints = []
        self.mimic_joints = []
        
        for joint_name in all_joints:
            if joint_name in self.mimic_config:
                self.mimic_joints.append(joint_name)
            else:
                self.independent_joints.append(joint_name)
        
        n_independent = len(self.independent_joints)
        
        if n_independent == 0:
            print("警告: 没有找到独立关节")
            return
        
        # 创建映射矩阵 (n_joints x n_independent)
        self.mimic_matrix = torch.zeros(n_joints, n_independent, dtype=self.dtype, device=self.device)
        
        # 创建offset向量 (n_joints,)
        self.mimic_offset = torch.zeros(n_joints, dtype=self.dtype, device=self.device)
        
        # 填充独立关节的单位映射和mimic关节的映射关系
        for i, joint_name in enumerate(all_joints):
            if joint_name in self.independent_joints:
                # 独立关节的单位映射
                ind_idx = self.independent_joints.index(joint_name)
                self.mimic_matrix[i, ind_idx] = 1.0
            elif joint_name in self.mimic_config:
                # mimic关节的映射关系
                config = self.mimic_config[joint_name]
                parent_joint = config['parent']
                multiplier = config.get('multiplier', 1.0)
                offset = config.get('offset', 0.0)
                
                if parent_joint in self.independent_joints:
                    parent_idx = self.independent_joints.index(parent_joint)
                    self.mimic_matrix[i, parent_idx] = multiplier
                    self.mimic_offset[i] = offset
                else:
                    print(f"警告: mimic关节 {joint_name} 的父关节 {parent_joint} 不在独立关节列表中")
    
    def expand_joint_values(self, th_independent):
        """
        将独立关节值扩展为全部关节值
        
        Args:
            th_independent: 独立关节的值，形状为 (batch_size, n_independent) 或 (n_independent,)
        
        Returns:
            th_full: 所有关节的值，形状为 (batch_size, n_joints)
        """
        th_independent = self.ensure_tensor(th_independent)
        th_independent = torch.atleast_2d(th_independent)
        
        # 应用映射矩阵: (batch_size, n_independent) @ (n_independent, n_joints)^T
        th_full = th_independent @ self.mimic_matrix.T
        
        # 一次性添加所有offset
        th_full = th_full + self.mimic_offset.unsqueeze(0)
        
        return th_full
    
    def forward_kinematics(self, th_independent, end_only: bool = True):
        """
        使用独立关节值计算正向运动学
        
        Args:
            th: 关节值 - 可以是全部关节(与Chain兼容)或独立关节
            end_only: 是否只计算末端执行器的变换。不处理，仅为了符合pv.RobotSDF接口。

        Returns:
            变换字典
        """
        th_full = self.expand_joint_values(th_independent)
        th_full = self.clamp(th_full)
        return super().forward_kinematics(th_full, frame_indices=None)
    
    def get_independent_joint_names(self):
        """获取独立关节名称列表"""
        return self.independent_joints
    
    def get_mimic_joint_names(self):
        """获取mimic关节名称列表"""
        return self.mimic_joints
    
    def get_joint_parameter_names(self, exclude_fixed=True):
        """
        重写父类方法，只返回独立关节名称，确保与RobotSDF等外部组件兼容
        
        Args:
            exclude_fixed: 是否排除固定关节
            
        Returns:
            独立关节名称列表
        """
        if hasattr(self, 'independent_joints'):
            return self.independent_joints
        else:
            # 如果independent_joints未初始化，调用父类方法
            return super().get_joint_parameter_names(exclude_fixed)
    
    def get_all_joint_parameter_names(self, exclude_fixed=True):
        """
        获取所有关节名称（包括mimic关节），保留原始功能
        
        Args:
            exclude_fixed: 是否排除固定关节
            
        Returns:
            所有关节名称列表
        """
        return super().get_joint_parameter_names(exclude_fixed)
    
    def get_mimic_config(self):
        """获取mimic配置"""
        return self.mimic_config
    
    def print_mimic_info(self):
        """打印mimic关节信息"""
        print(f"总关节数量: {len(self.get_all_joint_parameter_names())}")
        print(f"独立关节数量: {len(self.independent_joints)}")
        print(f"Mimic关节数量: {len(self.mimic_joints)}")
        print(f"独立关节: {self.independent_joints}")
        print(f"Mimic关节: {self.mimic_joints}")
        
        if self.mimic_config:
            print("\nMimic关节配置:")
            for joint_name, config in self.mimic_config.items():
                print(f"  {joint_name} -> {config['parent']} (multiplier: {config['multiplier']}, offset: {config['offset']})")