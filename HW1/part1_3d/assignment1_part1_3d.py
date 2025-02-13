from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.configuration = conf
        self.parent = None
        self.child = set()

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.child.add(child)

def sample_conf():
    while True:
        joint_1 = np.random.uniform(-2*np.pi, 2*np.pi)
        joint_2 = np.random.uniform(-2*np.pi, 2*np.pi)
        joint_3 = np.random.uniform(-np.pi, np.pi)
        
        s_conf = (joint_1, joint_2, joint_3)
        
        if not collision_fn(s_conf):
            return s_conf
   
def find_nearest(rand_node, node_list):
    nearest_node = None
    distance = float("inf")

    for curr_node in node_list:
        curr_distance = np.linalg.norm(np.array(curr_node.configuration) - np.array(rand_node.configuration))
        
        if curr_distance < distance:
            nearest_node = curr_node
            distance = curr_distance

    return nearest_node

def steer_to(rand_node, nearest_node):
    step_size = 0.05
    rand_conf = np.array(rand_node.configuration)
    nearest_conf = np.array(nearest_node.configuration)

    direction = rand_conf - nearest_conf
    distance = np.linalg.norm(direction)

    if distance < step_size:
        return rand_node

    unit_vector = direction/distance
    step_vector = step_size * unit_vector

    new_conf = nearest_conf + step_vector
    res = RRT_Node(tuple(new_conf)) 
    
    return res


def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################
    # print("inside the RRT algorithm")
    vertices = set()
    
    # following the python LEGB rule, not sure if allowed to change RRT func params
    start = RRT_Node(start_conf)
    goal = RRT_Node(goal_conf)

    
    vertices.add(start)

    while True:
        sampled_configuration = sample_conf() #guaranteed to be valid
        random_node = RRT_Node(sampled_configuration)
        nearest_node = find_nearest(random_node, vertices)
        new_node = steer_to(random_node, nearest_node)
        if collision_fn(new_node.configuration):
            continue
        nearest_node.add_child(new_node)
        new_node.set_parent(nearest_node)
        vertices.add(new_node)

        if np.linalg.norm(np.array(new_node.configuration) - np.array(goal_conf)) < 0.23:
            goal.set_parent(new_node)
            new_node.add_child(goal)
            vertices.add(goal)
            break
        
    path = []
    temp = goal
    while temp != None:
        path.append(temp.configuration)
        temp = temp.parent
    
    path = path[::-1]
    path.append(goal_conf)
    return path


###############################################################################
#your implementation ends here

if __name__ == "__main__":
    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-62.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    
		# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    path_conf = RRT()

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)
                # print(f"{q=}")