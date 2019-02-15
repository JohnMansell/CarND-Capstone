#!/usr/bin/env python

import math
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import numpy as np
from std_msgs.msg import Int32, String
from scipy.spatial import KDTree



'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0

class WaypointUpdater(object):

    # =======================
    #       Initialize
    # =======================
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Declare Variables
        self.car_pose       = None
        self.base_waypoints = None
        self.lane           = Lane()
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.stopline_wp_idx = -1

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('obstacle_waypoint', , self.obstacle_cb)

        # Publisher
        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

        self.debug_publisher = rospy.Publisher('debug', String, queue_size = 1)

        # Keep Alive
        self.update()

    # ---------------------------------
    #       Get Closest Way Point
    # ---------------------------------
    def get_closest_waypoint_idx(self):
        """ find the index of the closest waypoint ahead of the car, in base_waypoints"""

        # Position
        x = self.car_pose.pose.position.x
        y = self.car_pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Coordinates
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Hyper Plane
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    # -----------------------
    #       Update
    # -----------------------
    def update(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():

            if self.base_waypoints and self.car_pose and self.waypoint_tree:
                closest_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(self.base_waypoints.waypoints[
                    closest_idx:closest_idx + LOOKAHEAD_WPS])
            rate.sleep()

    # -----------------------
    #       Publish
    # -----------------------
    def publish_waypoints(self, waypoints):
        self.lane.header = self.base_waypoints.header
        self.lane.waypoints = waypoints
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)



    # -----------------------
    #       Generate Lane
    # -----------------------
    def generate_lane(self):
        lane = Lane()

        # Slice Way Points
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx: farthest_idx]

        # Traffic Light Detected
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints

        # No Traffic Light Detected
        else:
            lane.waypoints = self.decelerate_wayponits(base_waypoints, closest_idx)

        return lane

    # -------------------------------
    #       Decelerate Way Points
    # -------------------------------
    def decelerate_wayponits(self, waypoints, closest_idx):

        # New Waypoints
        temp = []
        for i, wp in enumerate(waypoints):

            # Copy Base Waypoint
            p = Waypoint()
            p.pose = wp.pose

            # Distance to Stop
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)

            # Change Velocity
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    # ============================================
    #       Call Back Functions
    # ============================================
    # --------------------------
    #       Call Back -- Pose
    # --------------------------
    def pose_cb(self, msg):
        self.car_pose = msg

    # -------------------------------
    #       Call Back -- Way Points
    # -------------------------------
    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    # -----------------------------
    #       Call Back -- Traffic
    # -----------------------------
    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    # -----------------------------
    #       Call Back -- Obstacle
    # -----------------------------
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it
        # later
        pass

    # ------------------------------------------
    #       Call Back -- Get Waypoint Velocity
    # ------------------------------------------
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    # ------------------------------------------
    #       Call Back -- Set Waypoint Velocity
    # ------------------------------------------
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # ---------------------
    #       Distance
    # ---------------------
    def distance(self, waypoints, wp1, wp2):
        dist = 0

        def dl(a, b): return math.sqrt((a.x - b.x) **
                                       2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist


def distance_to_single_point(wayp1, wayp2):
    x, y, z = wayp1.x - wayp2.x, wayp1.y - wayp2.y, wayp1.z - wayp2.z
    return math.sqrt(x * x + y * y + z * z)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
