#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):

    # -------------------------------
    #       Init
    # -------------------------------
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('tl_detector')

        # Declare attributes to store the received ROS messages
        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.light_classifier = None
        # Time estimation for processing the classifier
        self.compdelta_t = 0    # estimate time for image frame
        self.comp_t = 0         # estimate time for classifier computing
        self.image_timestamp = rospy.get_time() # needed for estimating image frame time


        # Read the node configuration
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Declare traffic waypoint publisher
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Initialize helper objects
        self.bridge = CvBridge()

        # Use different models for site and simulation
        _config = yaml.load(config_string)

        self.light_classifier = TLClassifier(_config)


        self.listener = tf.TransformListener()

        # Declare internal variables to track the state of the node
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = False
        # Subscribe to the input topics
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        # Execute ROS node
        rospy.spin()

# =====================================================
#       Call Back Functions
# =====================================================
    # -------------------------------
    #       Call back -- Pose
    # -------------------------------
    def pose_cb(self, msg):
        self.pose = msg

        # TODO: Remove after full integration of the classifier
        if self.waypoint_tree:
            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

    # -------------------------------
    #       Call Back -- Way Points
    # -------------------------------
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        if not self.waypoint_tree:
            waypoint_list = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                             waypoints.waypoints]
            self.waypoint_tree = KDTree(waypoint_list)

    # -------------------------------
    #       Call Back -- Traffic
    # -------------------------------
    def traffic_cb(self, msg):
        self.lights = msg.lights

    # -------------------------------
    #       Call Back -- Image
    # -------------------------------
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        
        # # # TODO remove after testing **********************************
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #
        # self.compdelta_t =rospy.get_time()-self.image_timestamp # calc time duration for classification
        # if self.compdelta_t>self.comp_t :
        #     rospy.loginfo('Fixed Time Step {} '.format(self.compdelta_t))
        #     t = rospy.get_time()
        #     self.image_timestamp = t # reset time stamp
        #     # Calc computing time
        #
        #     # Classifiy
        #     state = self.light_classifier.get_classification(cv_image)
        #     self.comp_t = rospy.get_time()-t + 0.1 # estimate computing time + buffer of 100ms
        #     rospy.loginfo('Computing Time  {} '.format(self.comp_t))
        #     rospy.loginfo('State {} '.format(state))

        # # # Test _end ***************************************************


        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    # ------------------------------------------
    #       Call Back -- Get closest waypoint
    # -----------------------------------------
    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    # -------------------------------
    #       Get Light State
    # -------------------------------
    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # -------------------------------------------------
        # Use Keras to classify the traffic light state
        #       - Uncomment to test keras classifier
        # -------------------------------------------------
        if (not self.has_image):
           self.prev_light_loc = None
           return False
        #
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #

        # calc time duration for classification
        self.compdelta_t =rospy.get_time()-self.image_timestamp
        if self.compdelta_t>self.comp_t :
            rospy.loginfo('Fixed Time Step {} '.format(self.compdelta_t))
            t = rospy.get_time()
            self.image_timestamp = t # reset time stamp
            # Calc computing time

            # Classifiy
            state = self.light_classifier.get_classification(cv_image)
            self.comp_t = rospy.get_time()-t + 0.1 # estimate computing time + buffer of 100ms
            rospy.loginfo('Computing Time  {} '.format(self.comp_t))
            rospy.loginfo('State {} '.format(state))
            return state


        # Return the ground truth Value of the traffic light
        return light.state


    # -------------------------------
    #       Process Traffic Lights
    # -------------------------------
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Declare variables to store the closest light and line waypoint index
        closest_light = None
        line_wp_idx = None

        # Get a list of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # If localization data is available find the closest wapoint
        if self.pose:
            # Find the closest waypoint to the car
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # Find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]

                # Get temporary waypoint index
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx

                # Only consider traffic lights in front of us
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        # If visible traffic light exists, publish its index waypoint index and state
        if closest_light:
            state = self.get_light_state(closest_light)
            
            return line_wp_idx, state

        # Otherwise publish a default message
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
