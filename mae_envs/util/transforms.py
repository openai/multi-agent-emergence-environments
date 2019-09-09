import numpy as np
from collections import OrderedDict
from mujoco_worldgen.transforms import closure_transform


def add_weld_equality_constraint_transform(name, body_name1, body_name2):
    '''
        Creates a weld constraint that maintains relative position and orientation between
        two objects
    '''
    def fun(xml_dict):
        if 'equality' not in xml_dict:
            xml_dict['equality'] = OrderedDict()
            xml_dict['equality']['weld'] = []
        constraint = OrderedDict()
        constraint['@name'] = name
        constraint['@body1'] = body_name1
        constraint['@body2'] = body_name2
        constraint['@active'] = False
        xml_dict['equality']['weld'].append(constraint)
        return xml_dict

    return fun


def set_joint_damping_transform(damping, joint_name):
    ''' Set joints damping to a single value.
        Args:
            damping (float): damping to set
            joint_name (string): partial name of joint. Any joint with joint_name
                as a substring will be affected.
    '''
    def closure(node):
        for joint in node.get('joint', []):
            if joint_name in joint['@name']:
                joint['@damping'] = damping
    return closure_transform(closure)


def remove_hinge_axis_transform(axis):
    ''' Removes specific hinge axis from the body. '''
    def fun(xml_dict):
        def closure(node):
            if 'joint' in node:
                node["joint"] = [j for j in node["joint"]
                                 if j["@type"] != "hinge"
                                 or np.linalg.norm(j["@axis"] - axis) >= 1e-5]
        return closure_transform(closure)(xml_dict)
    return fun
