import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import dlib
import time

showThresholds = True
drawOnlyImportantInformation = True
showImage = True

dir_path = "/home/nels/Desktop/ComputerVision4Melson/src/hand_signal_detector/resource/"
detector_name = 'Hand_Detector.svm'


class MyNode(Node):
    detector: None
    scale_factor : float
    size : int
    center_x : int
    center_y : int
    frame_counter :int
    result : int
    prev_result: int
    
    detection_counter = 0
    
    class Threshold():
        init = False
        right_x : int
        left_x : int
        middle_y : int
        size_up : int
        size_down : int
        def __init__(self):
            self.init = False
        
        def set(self, frame):
            # Thresholds
            self.right_x = int(frame.shape[1] * 2 / 3 )
            self.left_x = int(frame.shape[1] * 1 / 3)
            self.middle_y = int(frame.shape[0] / 2)

            self.size_up = 50000
            self.size_down = 10000
    
    def __init__(self):
        super().__init__('HandSignalDetector')
        self.init_hand_detector()
        self.image_sub = self.create_subscription(Image, 'image', self.listener_callback, 10)
        
        self.hand_position_pub = self.create_publisher(Int8, 'Hand/Position', 10)

        self.threshold = self.Threshold()
        self.cv_bridge = CvBridge()
        self.hand_position_msg = Int8()

    def listener_callback(self, msg):
        """ Callback for image subscriber """
        bridge = CvBridge()

        image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        if not self.threshold.init:
            self.threshold.set(image)
    
        image = self.detect_hand(image)
        
        self.publish_msgs()
        
        if showImage:
            cv2.imshow("Output image", image)
            cv2.waitKey(1)
            
    def init_hand_detector(self):
        detectorFullPath = os.path.sep.join([dir_path, detector_name])
        self.detector = dlib.simple_object_detector(detectorFullPath)
        self.scale_factor = 2.0
        self.size, self.center_x, self.center_y = 0, 0, 0
        self.frame_counter = 0
        self.result = 0
        self.prev_result = 0
        
    def detect_hand(self, frame):
        fps = 0
        start_time = time.time()    
        
        # Laterally flip the frame
        frame = cv2.flip( frame, 1 )
      
        # Calculate the Average FPS
        self.frame_counter += 1
        fps = (self.frame_counter / (time.time() - start_time))
        copy = frame.copy() 
        	     
        # Downsize the frame.
        new_width = int(frame.shape[1]/self.scale_factor)
        new_height = int(frame.shape[0]/self.scale_factor)
        resized_frame = cv2.resize(copy, (new_width, new_height))
        
        # Detect with detector
        detections = self.detector(resized_frame)
        
        self.prev_result = self.result
        self.result = 0
        text = "Brak detekcji"
        
        # Loop for each detection.
        for detection in (detections):   
            # Rescalling coordinates of detecttin
            x1 = int(detection.left() * self.scale_factor)
            y1 =  int(detection.top() * self.scale_factor)
            x2 =  int(detection.right() * self.scale_factor)
            y2 =  int(detection.bottom() * self.scale_factor)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if not drawOnlyImportantInformation:
                cv2.putText(frame, 'Hand Detected', (x1, y2+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)
	     
            # Calculate size of the hand.
            self.size = int((x2 - x1) * (y2 - y1))
	             
            # Extract the center of the hand on x-axis.
            self.center_x = (x2 + x1) // 2
            self.center_y = (y2 + y1) // 2
		    
            if self.center_x < self.threshold.left_x:
                if self.center_y < self.threshold.middle_y:
                    text = "Gora lewy"
                    self.result = 1
                else:
                    text = "Dol lewy"
                    self.result = 3
            elif self.center_x > self.threshold.right_x:
                if self.center_y < self.threshold.middle_y:
                    text = "Gora prawy"
                    self.result = 2
                else:
                    text = "Dol prawy"
                    self.result = 4
            else:
                if self.size > self.threshold.size_up:
                    text = "Duze"
                    self.result = 6
                else:
                    text = "Srodek"
                    self.result = 5
        
        # Latching result
        if self.result != 0:
            self.detection_counter += 5
            if self.detection_counter > 20:
                self.detection_counter = 20
        else:
            if self.detection_counter > 0:
                self.result = self.prev_result
                
            self.detection_counter -= 1
            if self.detection_counter < 0:
                self.detection_counter = 0
        
                    
        if not drawOnlyImportantInformation and self.showThresholds:
	        # Threshold marking
            cv2.line(frame, (self.threshold.left_x,0),(self.threshold.left_x, frame.shape[0]),(25,25,255), 1)
            cv2.line(frame, (self.threshold.right_x,0),(self.threshold.right_x, frame.shape[0]),(25,25,255), 1)
            cv2.line(frame, (0,self.threshold.middle_y),(frame.shape[1], self.threshold.middle_y),(25,25,255), 1)   
	    
        if not drawOnlyImportantInformation:
            # Display FPS and size of hand
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),1)

            # Computed information about hands
            cv2.putText(frame, f"Center: ({self.center_x},{self.center_y})", (400, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (233, 100, 25))
            	     
        cv2.putText(frame, text, (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (33, 100, 185), 2)

        return frame
      
    def publish_msgs(self):
        """ Publishing msgs """
        self.hand_position_msg.data = self.result
        self.hand_position_pub.publish(self.hand_position_msg)
            
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

