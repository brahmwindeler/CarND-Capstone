#!/usr/bin/env python
import rospy
import tf
import cv2
import yaml
import os
import math
import tensorflow
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.traffic_light_classifier import TrafficLightClassifier
from tl_helper import create_dir_if_nonexistent
from os.path import expanduser, join, exists
from kdtree import KDTree
from waypoint_helper import is_ahead
from waypoint_helper import get_simple_distance_from_waypoint

from ab_tl_classify.tl_classifier import TLClassifier
from ab_tl_detect.tl_detection import TLDetection
import PIL

# GLOBALS
STATE_COUNT_THRESHOLD = 3
TL_NEARNESS_THRESHOLD = 150

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector', log_level=rospy.INFO)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        #param_names = rospy.get_param_names()
        #rospy.loginfo("param_names: {}".format(param_names))

        self.PREFER_GROUND_TRUTH = rospy.get_param('/tl_PREFER_GROUND_TRUTH', False)

        # possible values: minotauro, ab2005
        self.detection_method = rospy.get_param('/tl_detection_method', 'minotauro')

        rospy.loginfo("Detection method: {}".format(self.detection_method))
        rospy.loginfo("Prefer ground truth? {}".format("True" if self.PREFER_GROUND_TRUTH else "False"))

        if self.detection_method == 'ab2005':
            self._models_initialized = False

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

        rospy.Subscriber('/image_color', Image, self.collect_images_callback)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.tl_state_pub = rospy.Publisher('/tl_state', Int32, queue_size=2)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Parameters for collecting frames from the camera
        self.should_collect_data = False
        self.dump_images_dir = create_dir_if_nonexistent(join(expanduser('~'), 'traffic_light_dataset', 'raw_images'))
        self.dump_images_counter = len(os.listdir(self.dump_images_dir))
        self.last_dump_tstamp = rospy.get_time()

        # Used to find the closest waypoint
        self.kdtree = None
        # Data file to store the image name and light state in the image.
        self.datafile = open(self.dump_images_dir + "/lightsData.csv", "w+")

        self.light_color = None

        if not self.PREFER_GROUND_TRUTH:
            # Create tensorflow session
            self.session = tensorflow.Session()

            # Initialize models
            if self.detection_method == 'ab2005':
                self.traffic_light_detector = TLDetection()
                self.light_classifier = TLClassifier()
                self._models_initialized = True
            else:
                # Import classifier and restore pre-trained weights
                self.light_classifier = TrafficLightClassifier(input_shape=[64, 64], learning_rate=1e-4)
                tensorflow.train.Saver().restore(self.session, TrafficLightClassifier.checkpoint_path)

        rospy.spin()

    def collect_images_callback(self, msg):
        """
        Save camera images (currently once per second)
        """
        def should_collect_camera_image():
            return self.should_collect_data and (rospy.get_time() - self.last_dump_tstamp > 1)

        if should_collect_camera_image():

            # Convert image message to actual numpy data
            image_data = self.bridge.imgmsg_to_cv2(msg)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)  # opencv uses BGR convention
            image_path = join(self.dump_images_dir, '{:06d}.jpg'.format(self.dump_images_counter))

            # Dump image to dump directory
            cv2.imwrite(image_path, image_data)

            # write the state of the light and the image name to a csv file
            rospy.logdebug("Writing to datafile")
            self.datafile.write('{:06d}.jpg'.format(self.dump_images_counter) + " , " +  self.light_color + "\n")
            # Update counter and timestamp
            self.dump_images_counter += 1
            self.last_dump_tstamp = rospy.get_time()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
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

        self.tl_state_pub.publish(Int32(state))

        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints       """

        if self.waypoints is not None and self.kdtree is None:

            rospy.logdebug('tl_detector: g_cl_wp: initializing kdtree')

            points = []

            for i, waypoint in enumerate(self.waypoints):
                points.append((float(waypoint.pose.pose.position.x),
                               float(waypoint.pose.pose.position.y),
                               i))

            self.kdtree = KDTree(points)

        if self.kdtree is not None:
            current_position = (pose.position.x, pose.position.y)
            closest = self.kdtree.closest_point(current_position)

            rospy.logdebug('tl_detector: g_cl_wp: closest point to {} is {}'.format(current_position, closest))

            return closest[2]

        return -1

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        light_state = TrafficLight.UNKNOWN

        if self.detection_method == 'ab2005':

            # Don't move until all systems are online
            if (not self._models_initialized):
                return TrafficLight.RED

            # Fix camera encoding to match model (from BGR to RGB)
            if hasattr(self.camera_image, 'encoding'):
                self.attribute = self.camera_image.encoding
                if self.camera_image.encoding == '8UC3':
                    self.camera_image.encoding = "rgb8"
            else:
                self.camera_image.encoding = 'rgb8'

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

            image = PIL.Image.fromarray(cv_image)

            # Detect traffic lights
            traffic_lights = self.traffic_light_detector.detect_traffic_lights(image)

            if len(traffic_lights) == 0:
                return TrafficLight.UNKNOWN

            # Classify detected traffic lights

            #start_time = rospy.get_time()
            light_state = self.light_classifier.get_classification(traffic_lights)
            #rospy.loginfo("get_light_state: classification elapsed time: {}, state: {}".format(rospy.get_time() - start_time, self._light_color(light_state)))

        else:

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            #start_time = rospy.get_time()
            light_state = self.light_classifier.get_classification(self.session, cv_image)
            #rospy.loginfo("get_light_state: classification elapsed time: {}, state: {}".format(rospy.get_time() - start_time, self._light_color(light_state)))

        if light_state != TrafficLight.UNKNOWN:
            rospy.loginfo("Traffic light detected. Color: {}".format(self._light_color(light_state)))

        return light_state

    def create_stop_line_pose(self, x, y, z):
        """takes in 3 positions and generates a traffic light object.
        Hijacked the light object and used it for a line to save time creating code
        """
        line = TrafficLight()
        line.pose = PoseStamped()
        line.pose.pose.position.x = x
        line.pose.pose.position.y = y
        line.pose.pose.position.z = z
        return line

    def get_closest_index(self, waypoint_idx, waypoint_list):
        """
        takes in the target waypoint index and a list of points.
        returns the index of the waypoint in the list closest to the target waypoint
        """
        target = self.waypoints[waypoint_idx]

        min_distance = float("Infinity") # drove the car around in manual, none of the distances for the lines were more than 25 away.

        for i, candidate in enumerate(waypoint_list):

            #rospy.logerr("iteration # " + str(int(i)))
            #distance = (candidate.pose.pose.position.x - target.pose.pose.position.x)**2 + (candidate.pose.pose.position.y - target.pose.pose.position.y)**2
            distance = get_simple_distance_from_waypoint(candidate, target.pose.pose)

            if distance < min_distance:
                #rospy.logerr("distance = " +str(float(distance)))
                min_distance = distance
                index = i

        return index

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        #if self.pose:
        #    car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        rospy.logdebug("tl_detector: p_tl: There are {} traffic lights to analyze.".format(len(self.lights)))

        min_distance = float("Infinity")
        for current_light in self.lights:

            # Check to see whether the traffic light is ahead of the car
            if is_ahead(current_light, self.pose.pose):

                # Get the simplified Euclidean distance (no sqrt) between it and the car
                light_distance = get_simple_distance_from_waypoint(current_light, self.pose.pose)

                # If the light is closer, remember it
                if (light_distance < min_distance):
                    min_distance = light_distance
                    light = current_light

        # If we found a light ahead of us
        if light:

            # Convert the light state to a human readible string
            self.light_color = self._light_color(light.state)

            # Calculate the actual distance the of the light.
            light_distance = math.sqrt(min_distance)

            rospy.logdebug("tl_detector: p_tl: closest light to {} is at {} (Distance: {}).".format(
                (self.pose.pose.position.x, self.pose.pose.position.y),
                (light.pose.pose.position.x, light.pose.pose.position.y),
                light_distance))

            # Convert the stop line positions to posestamped objects
            stop_line_list = []
            for stop_line_position in stop_line_positions:
                stop_line_pose = self.create_stop_line_pose(stop_line_position[0], stop_line_position[1], 0.0)
                stop_line_list.append(stop_line_pose)

            # rospy.logerr("List of Lines : " + str(stop_line_list))
            # Look up the closest waypoint to it
            # TODO: [brahm] Can we assume self.kdtree is initialized?

            # Instead of returning the index of the closest traffic light,
            # we will return the index of the closest stop line to the light

            # First we get the waypoint closest to the light
            light_wp_idx = self.get_closest_waypoint(light.pose.pose)

            # If we can't determine the closest waypoint, return an error
            if light_wp_idx < 0:
                rospy.logwarn("Can't determine closest waypoint to traffic light!")
                return -1, TrafficLight.UNKNOWN

            # Once we have the light waypoint index, use the same method to get the index of the closest stop line
            #rospy.loginfo('Closest index: {}, Stop Line list len: {}'.format(light_wp_idx, len(stop_line_list)))
            line_index = self.get_closest_index(light_wp_idx, stop_line_list)

            # Now that we have the stop line index from the list, we need to get the closest waypoint from our list
            light_wp_idx = self.get_closest_waypoint(stop_line_list[line_index].pose.pose)

            # Again, if we can't determine the closest waypoint, return an error
            if light_wp_idx < 0:
                rospy.logwarn("Can't determine closest waypoint to stop line!")
                return -1, TrafficLight.UNKNOWN

            # Determine the state of the light
            state = -1
            if self.PREFER_GROUND_TRUTH:

                rospy.logdebug("tl_detector: p_tl: Ground truth light color: {}".format(self._light_color(light.state)))

                # TODO: [brahm] Determine what light.state is when not available (e.g. not in the simulator)
                if light.state is not None:
                    state = light.state

            # If we don't know the state of the light yet, attempt to classify it
            if state == -1:
                state = self.get_light_state(light)

            # If the traffic light is close, let us know
            if (light_distance < TL_NEARNESS_THRESHOLD):
                rospy.logdebug("tl_detector: p_tl: light is close: {} meters away.".format(light_distance))

            return light_wp_idx, state

        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    # Helper
    def _light_color(self, state):
        if state == TrafficLight.RED:
            return "RED"
        elif state == TrafficLight.YELLOW:
            return "YELLOW"
        elif state == TrafficLight.GREEN:
            return "GREEN"
        else:
            return "UNKNOWN"

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
