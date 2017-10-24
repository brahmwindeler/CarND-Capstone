#!/usr/bin/env python

import os
import csv
import math
import yaml

from nav_msgs.msg import Path
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, PointStamped
from visualization_msgs.msg import MarkerArray, Marker
from styx_msgs.msg import TrafficLightArray, TrafficLight

import tf
import rospy

# Globals
CSV_HEADER = ['x', 'y', 'z', 'yaw']

class PathPublisher(object):

    def __init__(self):
        rospy.init_node('path_publisher', log_level=rospy.DEBUG)

        self.lights = None

        sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        self.pub_path   = rospy.Publisher('/nav_path', Path, queue_size=1, latch=True)
        self.pub_lights = rospy.Publisher('/nav_traffic_lights', MarkerArray, queue_size=1)
        self.pub_lines  = rospy.Publisher('/nav_stop_lines', MarkerArray, queue_size=1, latch=True)

        
        self.path_wp_publisher(rospy.get_param('~path', '../data/wp_yaw_const.csv'))

        self.stop_line_publisher(rospy.get_param('/traffic_light_config'))

        rospy.spin()

    def path_wp_publisher(self, path_file):
        if os.path.isfile(path_file):
            poses = self.load_poses(path_file)
            self.publish(poses)
            rospy.loginfo('Path Waypoints Published: {}'.format(len(poses)))
        else:
            rospy.logerr('%s is not a file', path_file)

    def stop_line_publisher(self, config):

        self.tl_config = yaml.load(config)
        stop_line_positions = self.tl_config['stop_line_positions']

        stop_line_markers = []
        for i, position in enumerate(stop_line_positions):
            m = Marker()
            m.header.frame_id = '/world'
            m.header.stamp = rospy.Time(0)
            m.ns = "stop_lines"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(position[0])
            m.pose.position.y = float(position[1])
            m.pose.position.z = 0.1
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 3.0
            m.scale.y = 3.0
            m.scale.z = 3.0
            # Color: Orange
            m.color.a = 1.0 # Don't forget to set the alpha!
            m.color.r = 1.0
            m.color.g = 0.67
            m.color.b = 0.0
            
            stop_line_markers.append(m)

        marker_array = MarkerArray()
        marker_array.markers = stop_line_markers

        self.pub_lines.publish(marker_array)
        rospy.loginfo('Stop Lines Published: {}'.format(len(stop_line_markers)))

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def load_poses(self, fname):
        poses = []
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            for wp in reader:
                p = PoseStamped()
                p.pose.position.x = float(wp['x'])
                p.pose.position.y = float(wp['y'])
                p.pose.position.z = float(wp['z'])
                q = self.quaternion_from_yaw(float(wp['yaw']))
                p.pose.orientation = Quaternion(*q)

                poses.append(p)
        return poses

    def publish(self, poses):
        path = Path()
        path.header.frame_id = '/world'
        path.header.stamp = rospy.Time(0)

        path.poses = poses
        self.pub_path.publish(path)

    def traffic_cb(self, msg):

        #if self.lights != None:
            #return

        self.lights = msg.lights

        tl_markers = []
        for i,tl in enumerate(self.lights):
            m = Marker()
            m.header.frame_id = '/world'
            m.header.stamp = rospy.Time(0)
            m.ns = "traffic_lights"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = tl.pose.pose.position.x
            m.pose.position.y = tl.pose.pose.position.y
            m.pose.position.z = tl.pose.pose.position.z
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 4.0
            m.scale.y = 4.0
            m.scale.z = 4.0
            m.color.a = 1.0 # Don't forget to set the alpha!
            if tl.state is not None:
                if tl.state == TrafficLight.RED:
                    m.color.r = 1.0
                    m.color.g = 0.0
                    m.color.b = 0.0
                elif tl.state == TrafficLight.YELLOW:
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                elif tl.state == TrafficLight.GREEN:
                    m.color.r = 0.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                else:
                    # Grey
                    m.color.r = 0.5
                    m.color.g = 0.5
                    m.color.b = 0.5
            else:
                m.color.r = 0.5
                m.color.g = 0.5
                m.color.b = 0.5
            
            tl_markers.append(m)

        marker_array = MarkerArray()
        marker_array.markers = tl_markers

        self.pub_lights.publish(marker_array)
        rospy.loginfo('Traffic Lights Published: {}'.format(len(tl_markers)))

if __name__ == '__main__':
    try:
        PathPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start PathPublisher node.')
