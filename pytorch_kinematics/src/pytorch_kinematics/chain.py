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
        axes_expanded = self.axes.unsqueeze(0).repeat(B_th, 1, 1)

        # compute all joint transforms at once first
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)

        # Get the base transformation matrix, ensure it's at least (1,4,4) for broadcasting
        # self.base_transform is already a Transform3d object, on correct device/dtype
        _initial_world_accumulator_matrix = self.base_transform.get_matrix().clone()
        if _initial_world_accumulator_matrix.ndim == 2:
            _initial_world_accumulator_matrix = _initial_world_accumulator_matrix.unsqueeze(0)  # Shape (1,4,4)
        if self.B_base == 1 and B_th > 1:
            # If base transform is a single matrix, we need to expand it to match B_th
            _initial_world_accumulator_matrix = _initial_world_accumulator_matrix.repeat(B_th, 1, 1)

        # Batch compatibility check
        if self.B_base > 1 and B_th > 1 and self.B_base != B_th:
            raise ValueError(f"Batch size of base_transform ({self.B_base}) and th ({B_th}) must be equal or one of them must be 1 for broadcasting.")

        # frame_transforms will store the computed world-coordinate matrices (tensors) for memoization
        # It maps integer frame indices to (Batch, 4, 4) tensors. Batch size is max(self.B_base, B_th)
        frame_world_transforms_map = {}

        # Ensure frame_indices is a list of integer items for dictionary keys and self.parents_indices access
        # The input frame_indices is a tensor on self.device.
        frame_indices_list = [idx_tensor.item() for idx_tensor in frame_indices]

        for requested_frame_idx_item in frame_indices_list:
            # Path for the current requested_frame_idx_item
            path_to_requested_frame = self.parents_indices[requested_frame_idx_item]

            # Initialize current_fk_matrix for the start of this path.
            # It starts as the world base transform.
            current_path_transform_matrix = _initial_world_accumulator_matrix.clone()

            if th.requires_grad:
                if th.numel() > 0:
                    graph_connected_one = (th.view(-1)[0] - th.view(-1)[0].detach()) + 1.0
                    current_path_transform_matrix = current_path_transform_matrix * graph_connected_one

            # Iterate through the ancestors in the path for the current requested_frame_idx_item
            # The first ancestor_idx in path_to_requested_frame is the root (index 0).
            # For the root itself, current_path_transform_matrix (which is base_transform) is its world transform if root's offsets/joints are identity.
            # If root has offsets/joint, they should be applied.

            # Let's refine how current_path_transform_matrix is initialized before the ancestor loop.
            # It should represent the transform of the PARENT of the first element in the current segment of the path.
            # If the path starts from root (ancestor_idx = 0), parent_transform is _initial_world_accumulator_matrix.
            # If we are calculating for a frame and its path is [0, 1, 2],
            # for ancestor_idx = 0: parent_transform = _initial_world_accumulator_matrix. After processing 0, frame_world_transforms_map[0] is set.
            # for ancestor_idx = 1: parent_transform = frame_world_transforms_map[0]. After processing 1, frame_world_transforms_map[1] is set.
            # for ancestor_idx = 2: parent_transform = frame_world_transforms_map[1]. After processing 2, frame_world_transforms_map[2] is set.

            last_processed_ancestor_transform = _initial_world_accumulator_matrix.clone()
            if th.requires_grad and th.numel() > 0: # connect to graph if needed
                graph_connected_one = (th.view(-1)[0] - th.view(-1)[0].detach()) + 1.0
                last_processed_ancestor_transform = last_processed_ancestor_transform * graph_connected_one

            # Find the point where path_to_requested_frame diverges from already computed paths
            # or starts from root.
            start_from_idx = 0 # which element in path_to_requested_frame to start computation from
            for i, ancestor_idx_val_in_path in enumerate(path_to_requested_frame):
                if ancestor_idx_val_in_path.item() in frame_world_transforms_map:
                    last_processed_ancestor_transform = frame_world_transforms_map[ancestor_idx_val_in_path.item()]
                    start_from_idx = i + 1 # Next iteration should start from child of this already computed frame
                else:
                    break # This ancestor_idx_val_in_path and subsequent ones need to be computed

            # Accumulate transforms from the identified starting point
            for i in range(start_from_idx, len(path_to_requested_frame)):
                ancestor_idx_tensor = path_to_requested_frame[i]
                ancestor_idx_item = ancestor_idx_tensor.item() # Integer index

                # current_fk_matrix for this ancestor_idx_item starts from its parent's world transform
                current_fk_matrix = last_processed_ancestor_transform

                link_offset_i = self.link_offsets[ancestor_idx_item]
                if link_offset_i is not None:
                    current_fk_matrix = current_fk_matrix @ link_offset_i

                joint_offset_i = self.joint_offsets[ancestor_idx_item]
                if joint_offset_i is not None:
                    current_fk_matrix = current_fk_matrix @ joint_offset_i

                jnt_idx = self.joint_indices[ancestor_idx_item]
                jnt_type = self.joint_type_indices[ancestor_idx_item]

                if jnt_type != 0: # If not a fixed joint
                    # Ensure batch sizes are compatible for matmul:
                    # current_fk_matrix could be (self.B_base,4,4) or (max(self.B_base,B_th),4,4)
                    # jnt_transform could be (B_th,4,4)
                    # PyTorch matmul broadcasting: (J,N,M) @ (K,M,P) -> (JorK, N,P) if J or K is 1
                    # Here: (B_cur,4,4) @ (B_th,4,4) where B_cur is self.B_base or max(self.B_base,B_th)
                    # This requires B_cur == B_th or one of them is 1. This is handled by the initial check.
                    if jnt_type == 1: # Revolute
                        jnt_transform_for_ancestor = rev_jnt_transform[:, jnt_idx]
                        current_fk_matrix = current_fk_matrix @ jnt_transform_for_ancestor
                    elif jnt_type == 2: # Prismatic
                        jnt_transform_for_ancestor = pris_jnt_transform[:, jnt_idx]
                        current_fk_matrix = current_fk_matrix @ jnt_transform_for_ancestor

                frame_world_transforms_map[ancestor_idx_item] = current_fk_matrix
                last_processed_ancestor_transform = current_fk_matrix
            # After iterating through all ancestors for requested_frame_idx_item,
            # its transform is in frame_world_transforms_map[requested_frame_idx_item]
            # OR, if path_to_requested_frame was empty (e.g. asking for a non-existent frame, though parents_indices should handle this)
            # or if all its ancestors were already computed, it's in last_processed_ancestor_transform.
            # The map should always have the entry due to the loop structure.

        # Construct the output dictionary using only the originally requested frame indices
        output_transform_objects = {}
        for requested_frame_idx_tensor in frame_indices: # Iterate over the original input frame_indices
            idx_val = requested_frame_idx_tensor.item()
            if idx_val in frame_world_transforms_map:
                 output_transform_objects[self.idx_to_frame[idx_val]] = tf.Transform3d(
                    matrix=frame_world_transforms_map[idx_val],
                    device=self.device,
                    dtype=self.dtype
                )
            # else: # Should not happen if frame_indices are valid and parents_indices is correct
            #    pass # Or raise error, or handle case of root frame if it's 0 and path is empty
                       # If path_to_requested_frame is empty (e.g. for frame_idx 0 if it has no "parent" in parents_indices[0])
                       # then frame_world_transforms_map[0] would not be set by the inner loop.
                       # This logic assumes parents_indices[0] = [0] for root.
                       # If requested_frame_idx_item is 0 (root), and parents_indices[0] is [0],
                       # start_from_idx will be 0. Loop range(0,1) runs for i=0, ancestor_idx_tensor=0.
                       # current_fk_matrix = last_processed_ancestor_transform (which is base_transform * graph_one)
                       # then offsets and joint transforms for frame 0 are applied. This is correct.

        return output_transform_objects

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
            th = torch.ones([*elem_shape, self.n_joints], device=self.device, dtype=self.dtype) * torch.nan
            joint_names = self.get_joint_parameter_names()
            for joint_name, joint_position in th_dict.items():
                jnt_idx = joint_names.index(joint_name)
                th[..., jnt_idx] = joint_position
            if torch.any(torch.isnan(th)):
                msg = "Missing values for the following joints:\n"
                for joint_name, th_i in zip(self.get_joint_parameter_names(), th):
                    msg += joint_name + "\n"
                raise ValueError(msg)
        return th

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
