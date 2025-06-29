from .urdf_parser_py.urdf import URDF, Mesh, Cylinder, Box, Sphere
from . import frame
from . import chain
import torch
import pytorch_kinematics.transforms as tf

JOINT_TYPE_MAP = {'revolute':   'revolute',
                  'continuous': 'revolute',
                  'prismatic':  'prismatic',
                  'fixed':      'fixed'}


def _convert_transform(origin):
    if origin is None:
        return tf.Transform3d()
    else:
        rpy = torch.tensor(origin.rpy, dtype=torch.float32, device="cpu")
        return tf.Transform3d(rot=tf.quaternion_from_euler(rpy, "sxyz"), pos=origin.xyz)


def _convert_visual(visual):
    if visual is None or visual.geometry is None:
        return frame.Visual()
    else:
        v_tf = _convert_transform(visual.origin)
        if isinstance(visual.geometry, Mesh):
            g_type = "mesh"
            g_param = (visual.geometry.filename, visual.geometry.scale)
        elif isinstance(visual.geometry, Cylinder):
            g_type = "cylinder"
            g_param = (visual.geometry.radius, visual.geometry.length)
        elif isinstance(visual.geometry, Box):
            g_type = "box"
            g_param = visual.geometry.size
        elif isinstance(visual.geometry, Sphere):
            g_type = "sphere"
            g_param = visual.geometry.radius
        else:
            g_type = None
            g_param = None
        return frame.Visual(v_tf, g_type, g_param)


def _build_chain_recurse(root_frame, lmap, joints):
    children = []
    for j in joints:
        if j.parent == root_frame.link.name:
            try:
                limits = (j.limit.lower, j.limit.upper)
            except AttributeError:
                limits = None
            # URDF assumes symmetric velocity and effort limits
            try:
                velocity_limits = (-j.limit.velocity, j.limit.velocity)
            except AttributeError:
                velocity_limits = None
            try:
                effort_limits = (-j.limit.effort, j.limit.effort)
            except AttributeError:
                effort_limits = None
            child_frame = frame.Frame(j.child)
            child_frame.joint = frame.Joint(j.name, offset=_convert_transform(j.origin),
                                            joint_type=JOINT_TYPE_MAP[j.type], axis=j.axis, limits=limits,
                                            velocity_limits=velocity_limits, effort_limits=effort_limits)
            link = lmap[j.child]
            child_frame.link = frame.Link(link.name, offset=_convert_transform(link.origin),
                                          visuals=[_convert_visual(link.visual)])
            child_frame.children = _build_chain_recurse(child_frame, lmap, joints)
            children.append(child_frame)
    return children


def build_chain_from_urdf(data):
    """
    Build a Chain object from URDF data.

    Parameters
    ----------
    data : str
        URDF string data.

    Returns
    -------
    chain.Chain
        Chain object created from URDF.

    Example
    -------
    >>> import pytorch_kinematics as pk
    >>> data = '''<robot name="test_robot">
    ... <link name="link1" />
    ... <link name="link2" />
    ... <joint name="joint1" type="revolute">
    ...   <parent link="link1"/>
    ...   <child link="link2"/>
    ... </joint>
    ... </robot>'''
    >>> chain = pk.build_chain_from_urdf(data)
    >>> print(chain)
    link1_frame
     	link2_frame

    """
    robot = URDF.from_xml_string(data)
    lmap = robot.link_map
    joints = robot.joints
    n_joints = len(joints)
    has_root = [True for _ in range(len(joints))]
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            if joints[i].parent == joints[j].child:
                has_root[i] = False
            elif joints[j].parent == joints[i].child:
                has_root[j] = False
    for i in range(n_joints):
        if has_root[i]:
            root_link = lmap[joints[i].parent]
            break
    root_frame = frame.Frame(root_link.name)
    root_frame.joint = frame.Joint()
    root_frame.link = frame.Link(root_link.name, _convert_transform(root_link.origin),
                                 [_convert_visual(root_link.visual)])
    root_frame.children = _build_chain_recurse(root_frame, lmap, joints)
    return chain.Chain(root_frame)


def build_serial_chain_from_urdf(data, end_link_name, root_link_name=""):
    """
    Build a SerialChain object from urdf data.

    Parameters
    ----------
    data : str
        URDF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.SerialChain
        SerialChain object created from URDF.
    """
    urdf_chain = build_chain_from_urdf(data)
    return chain.SerialChain(urdf_chain, end_link_name, root_link_name or '')


def build_mimic_chain_from_urdf(data):
    """
    Build a MimicChain object from urdf data.

    Parameters
    ----------
    data : str
        URDF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.MimicChain
        MimicChain object created from URDF.
    """
    urdf_chain = build_chain_from_urdf(data)

    import xml.etree.ElementTree as ET
        
    if isinstance(data, bytes):
        data = data.decode('utf-8')
        
    # 解析XML
    root = ET.fromstring(data)
        
    mimic_config = {}
        
    # 查找所有带有mimic标签的关节
    for joint in root.findall('.//joint'):
        mimic_elem = joint.find('mimic')
        if mimic_elem is not None:
            joint_name = joint.get('name')
            parent_joint = mimic_elem.get('joint')
            multiplier = float(mimic_elem.get('multiplier', 1.0))
            offset = float(mimic_elem.get('offset', 0.0))
            
            mimic_config[joint_name] = {
                'parent': parent_joint,
                'multiplier': multiplier,
                'offset': offset
            }

    return chain.MimicChain(urdf_chain._root, mimic_config)