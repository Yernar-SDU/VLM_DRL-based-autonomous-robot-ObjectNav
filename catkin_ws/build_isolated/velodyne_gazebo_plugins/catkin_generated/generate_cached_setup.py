# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/noetic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/noetic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/devel_isolated/velodyne_description;/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/devel_isolated/realsense_gazebo_plugin;/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/devel_isolated/realsense2_description;/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/devel_isolated/multi_robot_scenario;/opt/ros/noetic'.split(';'):
        python_path = os.path.join(workspace, 'lib/python3/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/devel_isolated/velodyne_gazebo_plugins/env.sh')

output_filename = '/home/ai-lab/Downloads/DRL-robot-navigation_segway_imu_should_be_calibrated/DRL-robot-navigation/catkin_ws/build_isolated/velodyne_gazebo_plugins/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
