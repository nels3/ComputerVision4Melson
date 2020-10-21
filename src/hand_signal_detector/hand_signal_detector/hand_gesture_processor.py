import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import dlib
import time

# params that can be set from command
showAll = False
showThresholds = False
showImage = False

dir_path = "/home/nels/Desktop/ComputerVision4Melson/src/hand_signal_detector/resource/"
detector_name = 'Hand_Detector.svm'

gesture_matrix =    {1: {0 : "left up", 5 : "left up", 3: "left move up"}, 
                     3: {0 : "left down", 5 : "left down", 1: "left move down"},
                     2: {0 : "right up", 6 : "right up", 4: "right move up"},
                     4: {0 : "right down",6 : "right down", 4: "right move down"},
                     5: {1 : "left", 3: "left", 0 : "left"},
                     6: {2 : "right", 4: "right", 0: "right"}}
                  

class MyNode(Node):
    prev_position = 0
    last_detected = 0
    prev_detected = 0
    
    hand_normal_position = [1, 2, 3, 4, 5, 6]
    hand_delete = [7]
    hand_accept = [8]
    
    gesture_accepted = False 
    gesture = "start"
    start_accept_timestamp = None
    start_delete_timestamp = None
    start_gesture_timestamp = None
  
    robot_ready = False
    
    def __init__(self):
        super().__init__('HandSignalDetector')       
       
        self.hand_position_sub = self.create_subscription(Int8, 'Hand/Position', self.listener_callback, 10)
        self.robot_ready_sub = self.create_subscription(Int8, 'Robot/Ready', self.robot_ready_callback, 10)
        self.hand_gesture_pub = self.create_publisher(String, 'Hand/Gesture', 10)
        
        self.hand_gesture_msg = String()
        self.start_gesture_timestamp = self.get_clock().now()
    
    def robot_ready_callback(self, msg):
        self.robot_ready = False

    def listener_callback(self, msg):
        """ Callback for image subscriber """
        hand_position = msg.data
        
        if self.robot_ready:
            # Changing position variables
            if hand_position in self.hand_normal_position:
                if self.last_detected != hand_position:
                    self.prev_detected = self.last_detected
                self.last_detected = hand_position
                self.start_accept_timestamp = None
                self.get_logger().info(f"{hand_position}: Last: {self.last_detected} Prev: {self.prev_detected}")
                
            # Accepting gesture
            elif self.last_detected != 0 and hand_position in self.hand_accept:
                current_time = self.get_clock().now()
                if self.start_accept_timestamp is None:
                    self.start_accept_timestamp = current_time
                else:
                    if current_time.nanoseconds - self.start_accept_timestamp.nanoseconds > 3000000000:
                        self.start_gesture_timestamp = current_time
                        self.recognize_gesture()
                        
                self.get_logger().info(f"Accepting: Last: {self.last_detected} Prev: {self.prev_detected}")
                        
                
            # Deleting gesture
            elif hand_position in self.hand_delete:
                self.get_logger().info("Want to delete?")
                current_time = self.get_clock().now()
                if self.start_delete_timestamp is None:
                    self.start_delete_timestamp = current_time
                else:
                    if current_time.nanoseconds - self.start_delete_timestamp.nanoseconds > 3000000000:                        
                        self.prev_detected = 0
                        self.last_detected = 0
                        self.start_accept_timestamp = None
                        self.get_logger().info(f"Deleting!!!")
            

        elif self.start_gesture_timestamp is not None:
            current_time = self.get_clock().now()
            
            self.get_logger().info(f"Robot is not ready. Gesture: {self.gesture}")
            
            # TODO: change waiting to optionaly wait or normal signal from other node
            if current_time.nanoseconds -  self.start_gesture_timestamp.nanoseconds> 1000000000:
                self.robot_ready = True
                self.prev_detected = 0
                self.last_detected = 0
                self.get_logger().info(f"Robot ready.")
            
            self.publish_msgs()           
       
        
    def recognize_gesture(self):
        self.gesture = "not know"
        if self.last_detected in gesture_matrix:
            if self.prev_detected in gesture_matrix[self.last_detected]:
                self.gesture = gesture_matrix[self.last_detected][self.prev_detected]
        self.robot_ready = False
        
      
    def publish_msgs(self):
        """ Publishing msgs """
        self.hand_gesture_msg.data = self.gesture
        self.hand_gesture_pub.publish(self.hand_gesture_msg)
            
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

