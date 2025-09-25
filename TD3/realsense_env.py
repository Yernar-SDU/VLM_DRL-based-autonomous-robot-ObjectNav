#!/usr/bin/env python3
import math
import os
import random
import subprocess
import time
from os import path
from sensor_msgs.msg import Imu

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

###############################################
#          Adjustable Parameters
###############################################
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

###############################################
class GazeboEnv:
    """
    Environment returning:
      - Depth-based channel (1, 64, 64)
      - 7D array of scalars [prev_lin, prev_ang, last_lin, last_ang, dist2goal, angle2goal, min_laser].
    """

    def __init__(self):
        self.episode_num = 0
        self.environment_dim = ENVIRONMENT_DIM
        self.odom_x = 0
        self.odom_y = 0
        self.odom_yaw = 0.0

        self.goal_x = 1
        self.goal_y = 0.0

        # used for randomizing the environment
        self.upper = 5.0
        self.lower = -5.0

        # store the single normalized depth image
        self.normed_depth = None
        self.last_odom = None

        # For stuck detection
        self.stuck_counter = 0
        self.last_robot_x = None
        self.last_robot_y = None

        self.near_wall_counter = 0

        # Two-step action memory
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)

        # For progress reward
        self.prev_distance = None
        # For "wall" reward from the depth camera
        self.prev_wall_real = None
        self.min_wall_real = 10.0

        # New: Variables to track cumulative rotation
        self.cum_rotation = 0.0
        self.last_yaw = None

        # For setting model state
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"

        self.max_steps = 400
        self.current_step = 0
        # For IMU sensor
        # self.imu_sub = rospy.Subscriber("/imu/data", Imu, self.imu_callback, queue_size=1)
        # self.latest_imu_data = None # Variable to store the latest IMU reading
        
        # self.orientation_y = None
        # self.imu_data = None

        # Launch roscore
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")


        # Initialize rospy + Gazebo launch
        rospy.init_node("gym", anonymous=True)

        if LAUNCHFILE.startswith("/"):
            fullpath = LAUNCHFILE
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", LAUNCHFILE)
        if not path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched with", LAUNCHFILE)


        # After launching Gazebo
        subprocess.Popen([
            "roslaunch", "-p", port,
            "rtabmap_ros", "rtabmap.launch",
            "rtabmap_args:=--delete_db_on_start",
            "depth:=true", "rgb:=false", "scan:=false"
        ])
        print("RTAB-Map SLAM launched!")



        # ROS publishers / services
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.publisher_goal = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher_lin = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher_ang = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber(
            "/realsense_camera/depth/image_raw", 
            Image, 
            self.depth_callback, 
            queue_size=1
        )
        self.odom_sub = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)

        rospy.sleep(2.0)
        print("Environment ready!")

    def odom_callback(self, od_data):
        self.last_odom = od_data
        self.odom_x = od_data.pose.pose.position.x
        self.odom_y = od_data.pose.pose.position.y
        # Convert orientation to yaw
        orientation_q = od_data.pose.pose.orientation
        quat = Quaternion(orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z)
        _, _, yaw = quat.to_euler()
        
        # Update cumulative rotation
        if self.last_yaw is not None:
            # Compute smallest angular difference
            delta_yaw = abs((yaw - self.last_yaw + math.pi) % (2 * math.pi) - math.pi)
            self.cum_rotation += delta_yaw
        self.last_yaw = yaw
        
        self.odom_yaw = yaw

    # def imu_callback(self, imu_data):
    #         """
    #         Callback function for the IMU subscriber.
    #         Stores the latest IMU data and prints it to the terminal. # <-- Modified description
    #         """
            # self.latest_imu_data = imu_data

            # --- ADD THIS LINE ---
            # Print the entire received message to the terminal
            # print("--- Received IMU Data: ---")
            # #print(imu_data)
            # print("--------------------------")
            # --- END ADDED LINES ---

            # Optional: You can still access specific fields if needed
            # orientation_q = imu_data.orientation
            # angular_velocity = imu_data.angular_velocity
            # linear_acceleration = imu_data.linear_acceleration
            # self.orientation_y = imu_data.orientation.y


            # Orientation (quaternion)
            # qx = imu_data.orientation.x
            # qy = imu_data.orientation.y
            # qz = imu_data.orientation.z
            # qw = imu_data.orientation.w

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            # roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])

            # Linear acceleration
            # ax = imu_data.linear_acceleration.x
            # ay = imu_data.linear_acceleration.y
            # az = imu_data.linear_acceleration.z

            # Angular velocity
            # gx = imu_data.angular_velocity.x
            # gy = imu_data.angular_velocity.y
            # gz = imu_data.angular_velocity.z

            # Store full IMU data: [roll, pitch, yaw, ax, ay, az, gx, gy, gz]
            # self.imu_data = np.array([roll, pitch, yaw, ax, ay, az, gx, gy, gz])

        # Print the extracted values
            # print("orientation = ", orientation_q)
            # print("angular_velocity = ", angular_velocity)
            # print("linear_acceleration = ", linear_acceleration)

    def depth_callback(self, msg):
        # print('msg', msg)
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        if cv_depth is None:
            return

        valid_mask = (cv_depth > 0) & np.isfinite(cv_depth)
        if not np.any(valid_mask):
            return

        valid_depths = cv_depth[valid_mask]
        min_depth_val = np.percentile(valid_depths, 5)
        max_depth_val = np.percentile(valid_depths, 95)
        if max_depth_val - min_depth_val < 1e-3:
            max_depth_val = min_depth_val + 1e-3

        normed = (cv_depth - min_depth_val) / (max_depth_val - min_depth_val)
        normed = np.clip(normed, 0.0, 1.0)
        normed = np.power(normed, GAMMA_VALUE)  # gamma correction
        self.normed_depth = normed.astype(np.float32)
        # print('self.normed_depth', self.normed_depth)
        # Scaled depth for collision detection
        scaled_depth = cv_depth * DISTANCE_SCALE
        # print('scaled_depth', scaled_depth)
        h, w = cv_depth.shape
        slice_width = w // ENVIRONMENT_DIM
        dist_array = np.ones(ENVIRONMENT_DIM, dtype=np.float32) * (10.0 * DISTANCE_SCALE)

        for i in range(ENVIRONMENT_DIM):
            cstart = i * slice_width
            cend = w if (i == ENVIRONMENT_DIM - 1) else (i + 1) * slice_width
            chunk = scaled_depth[:, cstart:cend]
            m = (chunk > 0) & np.isfinite(chunk)
            if np.any(m):
                dist_array[i] = chunk[m].min()

        # Minimum real distance for collision checks
        mask_scaled = (scaled_depth > 0) & np.isfinite(scaled_depth)
        if np.any(mask_scaled):
            self.min_wall_real = np.min(scaled_depth[mask_scaled])
        else:
            self.min_wall_real = 10.0

        # Optional debug display
        debug_frame = (normed * 255).astype(np.uint8)
        cv2.putText(
            debug_frame,
            f"Closest dist (scaled): {self.min_wall_real:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        # cv2.imshow("Depth Debug (Grayscale)", debug_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     print("Q pressed: resetting environment.")
        #     self.reset()

        self.realsense_data = dist_array
        # print('self.realsense_data', self.realsense_data)

    @staticmethod
    def observe_collision(dist_array):
        # print('callback_ orinentation y', orientation_y)
        # Dist array is scaled by DISTANCE_SCALE
        scaled_collision_threshold = COLLISION_DIST * DISTANCE_SCALE
        min_dist = dist_array.min()
        if min_dist < scaled_collision_threshold:
            return True, True, min_dist
        # Extra collision check: y value < 0
        # if orientation_y < -0.5 or orientation_y > 0.5:
        #     return True, True, min_dist
        return False, False, min_dist

    def step(self, action):
        self.current_step += 1
        target = False
        done = False

        # Publish velocity
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Step the simulation
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            pass

        # Collision detection
        done, collision, min_dist_array = self.observe_collision(self.realsense_data, )

        # Extra collision check: dark blue range
        if np.any(self.realsense_data < BLUE_DISTANCE_THRESHOLD):
            print("[Env] Extremely close => reset!")
            done = True
            collision = True

        # Stuck detection
        if self.last_robot_x is None:
            self.last_robot_x = self.odom_x
            self.last_robot_y = self.odom_y

        dist_moved = np.linalg.norm([self.odom_x - self.last_robot_x, self.odom_y - self.last_robot_y])
        if dist_moved < STUCK_MOVEMENT_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_robot_x = self.odom_x
        self.last_robot_y = self.odom_y

        stuck = False
        if self.stuck_counter >= STUCK_STEPS:
            print("[Env] Robot stuck => reset")
            done = True
            stuck = True
            # for i in range(30):
            #     vel_cmd = Twist()
            #     vel_cmd.linear.x = 0
            #     vel_cmd.angular.z = -0.2
            #     self.vel_pub.publish(vel_cmd)
            #     self.publish_markers(action)

            #     # Step the simulation
            #     rospy.wait_for_service("/gazebo/unpause_physics")
            #     try:
            #         self.unpause()
            #     except rospy.ServiceException:
            #         pass
            #     time.sleep(TIME_DELTA)
            #     rospy.wait_for_service("/gazebo/pause_physics")
            #     try:
            #         self.pause()
            #     except rospy.ServiceException:
            #         pass

        # Near wall detection
        if min_dist_array < (1.0 * DISTANCE_SCALE):
            self.near_wall_counter += 1
        else:
            self.near_wall_counter = 0
        if self.near_wall_counter >= NEAR_WALL_STEPS:
            print("[Env] near wall => reset")
            done = True

        # Check goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        if self.prev_distance is None:
            self.prev_distance = distance
        if self.prev_wall_real is None:
            self.prev_wall_real = self.min_wall_real

        # self.step_count += 1
        # if done:
        #     self.step_count = 0
        # elif self.step_count > MAX_STEPS and done is False:
        #     done = True
        #     stuck = True
            # target = False
        if self.current_step >= self.max_steps:
            print("[Env] Max steps reached => ending episode")
            done = True
        # Compute reward
        final_r = 0
        angle = self.compute_angle_to_goal()
        if self.episode_num < 300_000:
            final_r = self.get_reward_phase_1(
                target, collision, stuck,
                action,
                distance, self.prev_distance,
                angle,
                self.min_wall_real, self.prev_wall_real
            )
        # else:
        #     final_r = self.get_reward_phase_2(
        #         target, collision, stuck,
        #         action,
        #         distance, self.prev_distance,
        #         angle,
        #         self.min_wall_real, self.prev_wall_real
        #     )
        # print(f"[DEBUG] done={done}, collision={collision}, stuck={stuck}, reward={final_r:.2f}")

        self.prev_distance = distance
        self.prev_wall_real = self.min_wall_real

        # Build observation
        self.prev_action = self.last_action
        self.last_action = np.array([action[0], action[1]], dtype=np.float32)

        dist2goal = distance
        angle2goal = angle
        min_laser = min_dist_array

        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            dist2goal,
            angle2goal,
            min_laser,
        ], dtype=np.float32)

        # single-channel (64x64) depth
        if self.normed_depth is not None:
            resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
            one_channel = resized[None, ...]  # shape (1,64,64)
        else:
            one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        return (one_channel, scalars), final_r, done, target

    def reset(self):
        self.current_step = 0
        self.episode_num += 1
        self.stuck_counter = 0
        self.last_robot_x = None
        self.last_robot_y = None
        self.near_wall_counter = 0

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_distance = None
        self.prev_wall_real = None
        self.min_wall_real = 10.0

        # Reset cumulative rotation and last_yaw as well
        self.cum_rotation = 0.0
        self.last_yaw = None

        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            pass

        # Randomize position
        angle = np.random.uniform(-math.pi, math.pi)
        quat = Quaternion.from_euler(0, 0, angle)
        x_ok = False
        x = 0
        y = 0
        while not x_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            x_ok = self.check_pos(x, y)

        st = self.set_self_state
        st.pose.position.x = x
        st.pose.position.y = y
        st.pose.position.z = 0
        st.pose.orientation.x = quat.x
        st.pose.orientation.y = quat.y
        st.pose.orientation.z = quat.z
        st.pose.orientation.w = quat.w
        self.set_state.publish(st)

        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = angle

        self.random_box()
        self.change_goal()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            pass

        rospy.sleep(0.2)

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        self.prev_distance = distance

        angle_to_goal = self.compute_angle_to_goal()
        min_laser = self.min_wall_real

        scalars = np.array([
            self.prev_action[0],
            self.prev_action[1],
            self.last_action[0],
            self.last_action[1],
            distance,
            angle_to_goal,
            min_laser,
        ], dtype=np.float32)

        # 1-channel depth
        if self.normed_depth is not None:
            resized = cv2.resize(self.normed_depth, (IMG_WIDTH, IMG_HEIGHT))
            one_channel = resized[None, ...]
        else:
            one_channel = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        return (one_channel, scalars)

    def check_pos(self, x, y):
        goal_ok = True

        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            goal_ok = False

        if -1.3 > x > -2.7 and 4.7 > y > -0.2:
            goal_ok = False

        if -0.3 > x > -4.2 and 2.7 > y > 1.3:
            goal_ok = False

        if -0.8 > x > -4.2 and -2.3 > y > -4.2:
            goal_ok = False

        if -1.3 > x > -3.7 and -0.8 > y > -2.7:
            goal_ok = False

        if 4.2 > x > 0.8 and -1.8 > y > -3.2:
            goal_ok = False

        if 4 > x > 2.5 and 0.7 > y > -3.2:
            goal_ok = False

        if 6.2 > x > 3.8 and -3.3 > y > -4.2:
            goal_ok = False

        if 4.2 > x > 1.3 and 3.7 > y > 1.5:
            goal_ok = False

        if -3.0 > x > -7.2 and 0.5 > y > -1.5:
            goal_ok = False

        if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
            goal_ok = False

         # Check collision with boxes
        for bx, by in getattr(self, "box_positions", []):
            if np.linalg.norm([x - bx, y - by]) < 1:
                goal_ok = False
                break

        return goal_ok

    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        ok = False
        while not ok:
            gx = self.odom_x + random.uniform(self.upper, self.lower)
            gy = self.odom_y + random.uniform(self.upper, self.lower)
            ok = self.check_pos(gx, gy)
            if ok:
                self.goal_x = gx
                self.goal_y = gy

    def random_box(self):
        self.box_positions = []  # reset list of box positions
        for i in range(4):
            name = f"cardboard_box_{i}"
            box_ok = False
            while not box_ok:
                xx = np.random.uniform(-6, 6)
                yy = np.random.uniform(-6, 6)
                box_ok = self.check_pos(xx, yy)
                dist_robot = np.linalg.norm([xx - self.odom_x, yy - self.odom_y])
                dist_goal = np.linalg.norm([xx - self.goal_x, yy - self.odom_y])
                if dist_robot < 1.5 or dist_goal < 1.5:
                    box_ok = False
            self.box_positions.append((xx, yy))
            st = ModelState()
            st.model_name = name
            st.pose.position.x = xx
            st.pose.position.y = yy
            st.pose.position.z = 0
            st.pose.orientation.x = 0
            st.pose.orientation.y = 0
            st.pose.orientation.z = 0
            st.pose.orientation.w = 1
            self.set_state.publish(st)

    def publish_markers(self, action):
        # Visual markers for debugging
        markerArray = MarkerArray()
        # Goal marker
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher_goal.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = Marker.CUBE
        marker2.action = Marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0
        marker2.color.b = 0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0
        markerArray2.markers.append(marker2)
        self.publisher_lin.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = Marker.CUBE
        marker3.action = Marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0
        marker3.color.b = 0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0
        markerArray3.markers.append(marker3)
        self.publisher_ang.publish(markerArray3)

    def compute_angle_to_goal(self):
        dx = self.goal_x - self.odom_x
        dy = self.goal_y - self.odom_y
        desired_angle = math.atan2(dy, dx)
        angle_to_goal = desired_angle - self.odom_yaw
        # Wrap to [-pi, pi]
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        return angle_to_goal

    def get_reward_phase_1(
        self,
        target,
        collision,
        stuck,
        action,
        distance,
        prev_distance,
        angle,
        current_wall_real,
        prev_wall_real
    ):
        """
        An improved reward function that:
          - Rewards progress (delta in distance to goal)
          - Gives angle bonus for facing the goal
          - Penalizes collisions/stuck
          - Adds spin penalty if spinning in place
          - Adds a penalty for excessive cumulative rotation
          - Adds a step penalty to encourage faster completion
        """
        SLIDE_TOLERANCE = 0.1  # Acceptable deviation from y=0
        SLIDE_PENALTY_COEFF = 5.0

        gp = 0.0  # goal progress
        # Large reward for reaching goal / large penalty for collision
        if target:
            return 300.0
        if collision or stuck:
            return -300.0

        # 1) Base motion term: reward for forward speed, penalty for high angular speed.
        base = (action[0] / 2) - (abs(action[1]) / 2)

        # 2) Progress reward: scaled difference in distance (positive if closer, negative if farther).
        progress_coeff = 1.0 # неге 3.0 агай? 
        progress = progress_coeff * (prev_distance - distance)

        # 3) Angle bonus: encourage facing the goal.
        alpha = 1.0
        angle_bonus = alpha * (1.0 - (abs(angle) / math.pi))

        # 4) Wall penalty: penalize moving closer to walls.
        wall_penalty = 0.0
        dist_decreased = (prev_wall_real - current_wall_real)
        if dist_decreased > 0:
            wall_penalty = -dist_decreased

        # 5) Spin penalty: discourage in-place rotation.
        spin_penalty = 0.0
        if abs(action[0]) < 0.02 and abs(action[1]) > 0.2:
            spin_penalty = -0.02

        # 6) Small time-step penalty.
        step_penalty = TIME_STEP_PENALTY

        # 7) Cumulative rotation penalty: penalize excessive turning.
        rotation_penalty = 0.0
        if self.cum_rotation > 1.0:
            rotation_penalty = -0.1 * self.cum_rotation
            # Optionally reset cumulative rotation after applying the penalty
            self.cum_rotation = 0.0


        # backward penalty
        backward_penalty = 0.0
        if action[0] < 0:
            backward_penalty = -1
        # 8) Lateral sliding penalty (penalize deviation from y = 0)
        # slide_penalty = 0.0
        # if abs(self.imu_data[1]) > SLIDE_TOLERANCE:
        #     slide_penalty = -SLIDE_PENALTY_COEFF * abs(self.imu_data[1])
        # In get_reward_phase_1
        action_smoothness_penalty = -0.1 * np.linalg.norm(action - self.last_action)


        total = (
            base
            + progress
            + angle_bonus
            + spin_penalty
            + step_penalty
            + rotation_penalty
            + action_smoothness_penalty
            # + wall_penalty 
            # + 2 * backward_penalty
        )

   

        return total


    def get_reward_phase_2(
            self,
            target,
            collision,
            stuck,
            action,
            distance,
            prev_distance,
            angle,
            current_wall_real,
            prev_wall_real
        ):
            # 1) Base motion term: reward for forward speed, penalty for high angular speed.
            base = (action[0] / 2) - (abs(action[1]) / 2)
            # 6) Small time-step penalty.
            step_penalty = TIME_STEP_PENALTY

            # Large reward for reaching goal / large penalty for collision
            if target:
                return 300.0
            if collision or stuck:
                return -300.0


            total = (
                2*base
                # + 3 * progress 
                # + angle_bonus
                # +3*spin_penalty
                +  step_penalty
                # + rotation_penalty
                # + wall_penalty 
                # + 2 * backward_penalty
            )

            return total
