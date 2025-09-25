#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
import math
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker, MarkerArray
from actionlib_msgs.msg import GoalStatus

MAP_TOPIC = "/map"
ODOM_TOPIC = "/r1/odom"        # <-- matches your rostopic list
CMD_VEL_TOPIC = "/r1/cmd_vel"  # <-- matches your rostopic list
MOVE_BASE_ACTION = "/move_base"
FRONTIER_MARKER_TOPIC = "/frontier_markers"

EXPLORATION_PERIOD = 10.0  # seconds between exploration cycles


class ExplorerNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('explorer_node', anonymous=False)
        rospy.loginfo("[explorer_node] started")

        # Subscribers
        self.map_sub = rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.marker_pub = rospy.Publisher(FRONTIER_MARKER_TOPIC, MarkerArray, queue_size=1)

        # Action client
        self.move_base_client = actionlib.SimpleActionClient(MOVE_BASE_ACTION, MoveBaseAction)
        rospy.loginfo("[explorer_node] waiting for move_base action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("[explorer_node] move_base action server not available after 10s. Exiting.")
            rospy.signal_shutdown("move_base not available")
            return
        rospy.loginfo("[explorer_node] move_base action server available")

        # Internal state
        self.map_data = None
        self.robot_pose = None
        self.is_navigating = False

        # Timer
        self.timer = rospy.Timer(rospy.Duration(EXPLORATION_PERIOD), self.explore)

    def map_callback(self, msg):
        self.map_data = msg

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def explore(self, event=None):
        if self.is_navigating or self.map_data is None or self.robot_pose is None:
            return

        frontiers = self.find_frontiers()
        rospy.loginfo("[explore] found %d frontiers", len(frontiers))
        self.publish_frontier_markers(frontiers)

        if not frontiers:
            rospy.loginfo("[explore] no frontiers found. Exploration complete.")
            return

        target = self.select_best_frontier(frontiers)
        if target:
            self.navigate_to_frontier(target)

    def find_frontiers(self):
        msg = self.map_data
        h, w = msg.info.height, msg.info.width
        try:
            arr = np.array(msg.data, dtype=np.int8).reshape((h, w))
        except Exception as e:
            rospy.logerr("[find_frontiers] reshape failed: %s", str(e))
            return []

        is_unknown = (arr == -1)
        is_free = (arr == 0)
        frontiers = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if is_free[y, x]:
                    neigh = is_unknown[y - 1:y + 2, x - 1:x + 2]
                    if np.any(neigh):
                        frontiers.append((x, y))
        return frontiers

    def select_best_frontier(self, frontiers):
        if not frontiers:
            return None

        ox = self.map_data.info.origin.position.x
        oy = self.map_data.info.origin.position.y
        res = self.map_data.info.resolution

        rx = self.robot_pose.position.x
        ry = self.robot_pose.position.y

        rx_idx = int((rx - ox) / res)
        ry_idx = int((ry - oy) / res)

        best, best_d = None, float('inf')
        for fx, fy in frontiers:
            d = math.hypot(fx - rx_idx, fy - ry_idx)
            if d < best_d:
                best_d, best = d, (fx, fy)

        if best is None:
            return None

        wx = best[0] * res + ox
        wy = best[1] * res + oy
        rospy.loginfo("[select] chosen frontier at (%.2f, %.2f)", wx, wy)
        return (wx, wy)

    def navigate_to_frontier(self, world_xy):
        wx, wy = world_xy
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = wx
        goal.target_pose.pose.position.y = wy
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo("[navigate] sending goal x=%.2f y=%.2f", wx, wy)
        self.is_navigating = True
        self.move_base_client.send_goal(goal, done_cb=self.done_callback)

    def done_callback(self, status, result):
        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("[done_callback] reached goal")
        elif status == GoalStatus.ABORTED:
            rospy.logwarn("[done_callback] aborted -> running recovery")
            self.recover_robot()
        else:
            rospy.logwarn("[done_callback] goal status: %s", str(status))
        self.is_navigating = False

    def recover_robot(self):
        rospy.loginfo("[recover] rotating in place to attempt recovery")
        twist = Twist()
        twist.angular.z = 0.6
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < 6.0 and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def publish_frontier_markers(self, frontiers):
        if self.map_data is None:
            return

        ma = MarkerArray()
        for i, (fx, fy) in enumerate(frontiers[:200]):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "frontiers"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            ox = self.map_data.info.origin.position.x
            oy = self.map_data.info.origin.position.y
            res = self.map_data.info.resolution
            m.pose.position.x = fx * res + ox
            m.pose.position.y = fy * res + oy
            m.pose.position.z = 0.1

            m.scale.x = m.scale.y = m.scale.z = max(0.05, res * 2.0)
            m.color.a = 0.8
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0

            ma.markers.append(m)

        self.marker_pub.publish(ma)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = ExplorerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
