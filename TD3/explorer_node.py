#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, Odometry
from cv_bridge import CvBridge
import cv2
import os
#!/usr/bin/env python3
import rospy
import actionlib
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from actionlib_msgs.msg import GoalStatus
import subprocess
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
import time
# Adjustable
MAP_TOPIC = "/rtabmap/grid_map"
ODOM_TOPIC = "r1/odom"  # No leading slash for consistency
MOVE_BASE_ACTION = "move_base"
FRONTIER_MARKER_TOPIC = "/frontier_markers"
EXPLORATION_PERIOD = 10.0  # seconds
LAUNCHFILE = "multi_robot_scenario.launch"


class ExplorerNode:
    def __init__(self):
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        rospy.sleep(2.0)

        # Launch Gazebo
        if LAUNCHFILE.startswith("/"):
            fullpath = LAUNCHFILE
        else:
            # Assuming assets folder is in the same directory as the script
            # This is a more robust way to find the file
            script_dir = os.path.dirname(os.path.realpath(__file__))
            fullpath = os.path.join(script_dir, "assets", LAUNCHFILE)
        
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched with", LAUNCHFILE)
        rospy.sleep(5.0)
        rospy.init_node('explorer_node', anonymous=False)

        # Parameters from launch
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.odom_topic = rospy.get_param("~odom_topic", "/r1/odom")
        self.camera_topic = rospy.get_param("~camera_topic", "/realsense_camera/color/image_raw")
        self.save_dir = rospy.get_param("~save_dir", "/tmp/exploration_data")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.camera_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        # Internal state
        self.frame_id = 0

        rospy.loginfo("[ExplorerNode] Initialization complete.")
        rospy.spin()

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[camera_callback] CV Bridge error: {e}")
            return

        # Save image for VLM training
        filename = os.path.join(self.save_dir, f"frame_{self.frame_id:05d}.png")
        cv2.imwrite(filename, cv_image)
        self.frame_id += 1
        rospy.loginfo_throttle(10, f"[camera_callback] Saved image {filename}")

    def map_callback(self, msg):
        # Optionally save map as a 2D occupancy grid image
        import numpy as np
        grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        grid_img = ((grid + 1) * 127).astype(np.uint8)  # Unknown=-1->0, Free=0->127, Occupied=100->254
        map_filename = os.path.join(self.save_dir, "map_latest.png")
        cv2.imwrite(map_filename, grid_img)
        rospy.loginfo_throttle(30, f"[map_callback] Saved map image {map_filename}")

    def odom_callback(self, msg):
        # You can use odometry to log robot pose if needed
        pose = msg.pose.pose
        rospy.loginfo_throttle(30, f"[odom_callback] Robot at x={pose.position.x:.2f}, y={pose.position.y:.2f}")

if __name__ == "__main__":
    try:
        ExplorerNode()
    except rospy.ROSInterruptException:
        pass
