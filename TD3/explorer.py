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

ENVIRONMENT_DIM = 20
TIME_DELTA = 0.1
MAX_STEPS = 600
GOAL_REACHED_DIST = 0.4
COLLISION_DIST = 0.35

STUCK_STEPS = 60
STUCK_MOVEMENT_THRESHOLD = 0.02

NEAR_WALL_STEPS = 40
DISTANCE_SCALE = 0.01
BLUE_DISTANCE_THRESHOLD = 0.15

WALL_DISTANCE_THRESHOLD = 1
TIME_STEP_PENALTY = -1  # small negative reward each step

# For the depth image shape
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Gamma correction
GAMMA_VALUE = 0.8

class ExplorerNode:
    def __init__(self):
        # Internal state
        self.map_data = None
        self.robot_pose = None
        self.is_navigating = False

        # --- Process Launching (No changes here, but be aware it can be fragile) ---
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        rospy.sleep(2.0)

        rospy.init_node('explorer_node', anonymous=False)
        rospy.loginfo("[explorer_node] started")
        rospy.loginfo("[init] ExplorerNode initialized, waiting for subsystems")
        
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

        # Launch RTAB-Map
        # subprocess.Popen([
        #     "roslaunch", "-p", port,
        #     "rtabmap_ros", "rtabmap.launch",
        #     "rtabmap_args:=--delete_db_on_start",
        #     "depth_topic:=/realsense_camera/depth/image_raw",
        #     "rgb_topic:=/realsense_camera/color/image_raw",
        #     "camera_info_topic:=/realsense_camera/color/camera_info",
        #     "odom_topic:=" + ODOM_TOPIC,
        #     "frame_id:=base_link",
        #     # "odom_frame_id:=r1/odom", 
        #     "approx_sync:=false", # Set to false if topics don't sync perfectly
        #     "wait_for_transform:=0.2"
        # ])
        # print("RTAB-Map SLAM launched!")
        # rospy.sleep(3.0)

        # # Launch Move Base
        # subprocess.Popen([
        #     "roslaunch", "-p", port,
        #     "multi_robot_scenario", "move_base.launch"
        # ])
        # print("Move Base launched!")
        
        self.wait_for_system_ready(timeout=60.0)

        # CV Bridge
        self.bridge = CvBridge()
        self.normed_depth = None
        self.min_wall_real = 10.0
        
        # ROS Publishers / Services
        # **FIX: Only one cmd_vel publisher is needed.**
        self.cmd_vel_pub = rospy.Publisher("r1/cmd_vel", Twist, queue_size=10)
        rospy.loginfo("[init] cmd_vel publisher created on topic 'r1/cmd_vel'")

        self.set_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.marker_pub = rospy.Publisher(FRONTIER_MARKER_TOPIC, MarkerArray, queue_size=10)
        
        # Gazebo services
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)


        # Subscribers
        self.depth_sub = rospy.Subscriber(
            "/realsense_camera/depth/image_raw", 
            Image, 
            self.depth_callback, 
            queue_size=1
        )
        self.odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
        self.map_sub = rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback, queue_size=1)
        rospy.loginfo(f"[init] Subscribed to {ODOM_TOPIC} and {MAP_TOPIC}")
        
        rospy.sleep(2.0)
    
        # Action client
        self.move_base_client = actionlib.SimpleActionClient(MOVE_BASE_ACTION, MoveBaseAction)
        rospy.loginfo(f"[explorer_node] waiting for '{MOVE_BASE_ACTION}' action server (15s)...")
        if not self.move_base_client.wait_for_server(rospy.Duration(15.0)):
            rospy.logerr(f"[{rospy.get_name()}] move_base action server not available. Shutting down.")
            rospy.signal_shutdown("Move Base not available")
            return
        
        rospy.loginfo("[explorer_node] move_base action server is available.")

        # Timer
        self.timer = rospy.Timer(rospy.Duration(EXPLORATION_PERIOD), self.explore)
        print("Environment is ready!")
        rospy.loginfo("[explorer_node] Initialization complete. Starting exploration loop.")

    # ... (depth_callback method is fine, no changes needed) ...
    def depth_callback(self, msg):
        rospy.logdebug("[depth_callback] depth image received")
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e: 
            rospy.logerr("[depth_callback] CV Bridge Error: %s", e)
            return
        
        if cv_depth is None or cv_depth.size == 0:
            return

        valid_mask = (cv_depth > 0) & np.isfinite(cv_depth)
        if not np.any(valid_mask):
            self.min_wall_real = 10.0
            return

        # Simplified and safer normalization
        valid_depths = cv_depth[valid_mask]
        min_d, max_d = np.percentile(valid_depths, [5, 95])
        
        if max_d - min_d < 1e-3:
            normed = np.zeros_like(cv_depth, dtype=np.float32)
        else:
            normed = (cv_depth - min_d) / (max_d - min_d)
            normed = np.clip(normed, 0.0, 1.0)
        
        normed = np.power(normed, GAMMA_VALUE)
        self.normed_depth = normed.astype(np.float32)
        
        # Real distance for collision checks
        self.min_wall_real = np.min(valid_depths) / 1000.0 # Assuming depth is in mm


    # ... (map_callback and odom_callback are fine) ...
    def map_callback(self, msg):
        if self.map_data is None:
            rospy.loginfo("[map_callback] First RTAB-Map grid map received")
        else:
            rospy.logdebug("[map_callback] New map update received")
        self.map_data = msg

    def odom_callback(self, msg):
        if self.robot_pose is None:
            rospy.loginfo("[odom_callback] First odometry received")
        self.robot_pose = msg.pose.pose


    def explore(self, event=None):
        rospy.loginfo("[explore] Exploration cycle triggered")
        if self.is_navigating:
            rospy.loginfo("[explore] Robot is currently navigating to a goal. Skipping cycle.")
            return
        if self.map_data is None:
            rospy.logwarn_throttle(10, "[explore] Map has not been received yet. Cannot find frontiers.")
            return
        if self.robot_pose is None:
            rospy.logwarn_throttle(10, "[explore] Odometry has not been received yet. Cannot find frontiers.")
            return

        rospy.loginfo("[explore] Searching for frontiers...")
        frontiers = self.find_frontiers()

        if not frontiers:
            rospy.logwarn("[explore] No frontiers found. Exploration may be complete or map is not detailed enough.")
            self.publish_frontier_markers([]) # Clear old markers
            return
        
        rospy.loginfo(f"[explore] Found {len(frontiers)} frontier points.")
        self.publish_frontier_markers(frontiers)

        target_world_coords = self.select_best_frontier(frontiers)
        if target_world_coords is None:
            rospy.logerr("[explore] Failed to select a valid frontier goal.")
            return
            
        self.navigate_to_frontier(target_world_coords)

    # ... (find_frontiers, select_best_frontier, navigate_to_frontier, done_callback are fine) ...
    def find_frontiers(self):
        msg = self.map_data
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int8).reshape((h, w))
        
        # An unknown cell (-1) is a frontier candidate if it has a free neighbor (0)
        is_unknown = (arr == -1)
        is_free = (arr == 0)

        # Erode the free space to avoid selecting frontiers right next to obstacles
        from scipy.ndimage import binary_erosion
        eroded_free = binary_erosion(is_free, iterations=2)

        frontiers = []
        # Find unknown cells that are adjacent to free space
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if is_unknown[y, x]:
                    # Check 8 neighbors
                    if np.any(eroded_free[y-1:y+2, x-1:x+2]):
                        frontiers.append((x, y))
        return frontiers

    def select_best_frontier(self, frontiers):
        if not frontiers:
            return None
        
        ox, oy = self.map_data.info.origin.position.x, self.map_data.info.origin.position.y
        res = self.map_data.info.resolution
        rx, ry = self.robot_pose.position.x, self.robot_pose.position.y
        
        robot_map_x = int((rx - ox) / res)
        robot_map_y = int((ry - oy) / res)

        best_frontier = None
        min_dist_sq = float('inf')

        for fx, fy in frontiers:
            dist_sq = (fx - robot_map_x)**2 + (fy - robot_map_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_frontier = (fx, fy)
        
        if best_frontier:
            rospy.loginfo(f"[select] Chosen frontier at map coordinates {best_frontier} with distance {math.sqrt(min_dist_sq):.2f} cells.")
            # Convert best frontier back to world coordinates
            world_x = best_frontier[0] * res + ox + res / 2.0
            world_y = best_frontier[1] * res + oy + res / 2.0
            return (world_x, world_y)
        return None

    def navigate_to_frontier(self, world_xy):
        wx, wy = world_xy
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = wx
        goal.target_pose.pose.position.y = wy
        goal.target_pose.pose.orientation.w = 1.0
        
        rospy.loginfo(f"[navigate] Sending goal: x={wx:.2f}, y={wy:.2f} in 'map' frame.")
        self.is_navigating = True
        self.move_base_client.send_goal(goal, done_cb=self.done_callback)

    def done_callback(self, status, result):
        self.is_navigating = False
        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("[navigate] Goal reached successfully.")
        elif status == GoalStatus.ABORTED:
            rospy.logwarn("[navigate] Goal was aborted. The robot may be stuck.")
            self.recover_robot()
        else:
            rospy.logwarn(f"[navigate] Goal finished with status: {GoalStatus.to_string(status)}")

    def recover_robot(self):
        rospy.loginfo("[recover] Attempting recovery by rotating in place.")
        twist = Twist()
        twist.angular.z = 0.5 # A moderate rotation speed
        
        # Rotate for 5 seconds
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 5.0 and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
            
        # Stop the robot
        self.cmd_vel_pub.publish(Twist()) # Publish a zero-velocity twist
        rospy.loginfo("[recover] Recovery rotation finished.")


    # ... (publish_frontier_markers is fine) ...
    def publish_frontier_markers(self, frontiers):
        ma = MarkerArray()
        # Action DELETEALL clears old markers
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "frontiers"
        m.id = -1
        m.action = Marker.DELETEALL
        ma.markers.append(m)

        ox, oy, res = self.map_data.info.origin.position.x, self.map_data.info.origin.position.y, self.map_data.info.resolution
        for i, (fx, fy) in enumerate(frontiers[:300]):  # Limit markers for performance
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "frontiers"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = fx * res + ox + res / 2.0
            m.pose.position.y = fy * res + oy + res / 2.0
            m.pose.position.z = 0
            m.scale.x = res
            m.scale.y = res
            m.scale.z = res
            m.color.a = 0.7
            m.color.r = 0.1
            m.color.g = 1.0
            m.color.b = 0.1
            m.lifetime = rospy.Duration(EXPLORATION_PERIOD * 1.5)
            ma.markers.append(m)
        self.marker_pub.publish(ma)

    def run(self):
        rospy.spin()

    def wait_for_system_ready(self, timeout=30.0):
        rospy.loginfo("Checking for system readiness (odom, map, tf)...")
        start_time = rospy.Time.now()
        rate = rospy.Rate(2)
        
        # TF Buffer must be created once and used
        try:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer)
        except ImportError:
            rospy.logerr("tf2_ros not found. Please install python3-tf2-ros.")
            return False

        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < timeout:
            odom_ready = self.robot_pose is not None
            map_ready = self.map_data is not None
            
            tf_ready = False
            try:
                # Check for a recent transform
                if self.tf_buffer.can_transform('map', 'base_link', rospy.Time(0), timeout=rospy.Duration(0.5)):
                    tf_ready = True
            except Exception as e:
                rospy.logdebug(f"TF check failed: {e}")
                tf_ready = False

            if odom_ready and map_ready and tf_ready:
                rospy.loginfo("System is ready: Odom, Map, and TF (map->base_link) are all available.")
                return True
            
            rospy.logdebug(f"Waiting for system: odom={odom_ready}, map={map_ready}, tf={tf_ready}")
            rate.sleep()
            
        rospy.logwarn(f"System readiness check timed out after {timeout} seconds.")
        return False


if __name__ == "__main__":
    try:
        node = ExplorerNode()
        if rospy.is_shutdown():
            print("Failed to initialize ExplorerNode properly.")
        else:
            node.run()
    except rospy.ROSInterruptException:
        pass
    except IOError as e:
        print(f"Error during initialization: {e}")

