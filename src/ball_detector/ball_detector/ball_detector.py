import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import imutils
from collections import deque

doCalibration = True
doTracking = True
showImage = True

# Inital value for ball color threshold
h_low = 23
s_low = 86
v_low = 47
h_high = 53
s_high = 255
v_high = 255

TRACKING_BUFFER = 20
FRAME_WIDTH = 600
MIN_RADIUS = 10
MAX_RADIUS = 100

tribes = ["hough", "contour"]
BALL_DETECTOR_TRIBE = tribes[1]

class MyNode(Node):    
    tribes_function_mapper = {'hough': (lambda MyNode, image: MyNode.find_ball_using_houghcircle(image)), 'contour': (lambda MyNode, image: MyNode.find_ball_using_contour(image))}
	
    initialized:bool
    do_calibration:bool
    do_tracking:bool
    show_image:bool
	
    hsv_low = None
    hsv_high = None
    pts = None
    
    class HSV_threshold():
        h:int
        s:int
        v:int
        def __init__(self, h, s, v):
            self.h = h
            self.s = s
            self.v = v
               
    def __init__(self):
        super().__init__('BallDetector')
        self.declare_parameter("do_calibration", doCalibration)
        self.declare_parameter("do_tracking", doTracking)
        self.declare_parameter("show_image", showImage)
        self.do_calibration = self.get_parameter("do_calibration").value
        self.do_tracking = self.get_parameter("do_tracking").value
        self.show_image = self.get_parameter("show_image").value
        
        self.initialized = False
        self.hsv_low = self.HSV_threshold(h_low, s_low, v_low)
        self.hsv_high = self.HSV_threshold(h_high, s_high, v_high)
        
        if self.do_tracking:
            self.pts = deque(maxlen=TRACKING_BUFFER)
			
        self.image_sub = self.create_subscription(Image, 'image', self.listener_callback, 10)
        
			
    def listener_callback(self, msg):
        """ Callback for image subscriber """
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.do_calibration:
            if not self.initialized:
                self.init_calibration()
            self.calibrate_ball_hsv_filter(image)
		
        image, center = self.tribes_function_mapper[BALL_DETECTOR_TRIBE](self, image)
        
        if self.show_image:
            cv2.imshow("Output image", image)
            cv2.waitKey(1)
	
    def process_tracking(self, image, center):
        if self.do_tracking:
            self.pts.appendleft(center)
            # loop over the set of tracked points
            for i in range(1, len(self.pts)):    
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue
                thickness = int(np.sqrt(TRACKING_BUFFER / float(i + 1)) * 2.5)
                cv2.line(image, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        return image
        
    def find_ball_using_contour(self, original):
        frame = imutils.resize(original, width=FRAME_WIDTH)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([self.hsv_low.h, self.hsv_low.s, self.hsv_low.v]),np.array([self.hsv_high.h, self.hsv_high.s, self.hsv_high.v]))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
		
        center = None
    
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            # only proceed if the radius meets a minimum size
            if radius > MIN_RADIUS and radius < MAX_RADIUS:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if self.show_image:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame,str(2*radius),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
            
        frame = self.process_tracking(frame, center)

        return frame, center
        
    def find_ball_using_houghcircle(self, original):
        frame = imutils.resize(original, width=FRAME_WIDTH)
        ball_mask = self.create_ball_mask(frame)
        original_blur = cv2.GaussianBlur(frame, (3, 3), 0)
        ball_image = cv2.bitwise_and(original_blur, original_blur, mask=ball_mask)
        gray = cv2.cvtColor(ball_image, cv2.COLOR_BGR2GRAY)
        
        if self.show_image:
    	    cv2.imshow("Gray", gray)
        
        canny_edge = cv2.Canny(gray, 50, 240)
        circles = cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=100, param2=30,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
							   
        center = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(frame, center, 1, (0, 100, 100), 3)
                radius = i[2]
                if self.show_image:
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame,str(2*radius),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
     
        frame = self.process_tracking(frame, center)      
        return frame, center
            
    
    def init_calibration(self):
        def nothing(x):
            pass
        cv2.namedWindow('HSV_Calibration_TrackBar')
        cv2.createTrackbar('H_low', 'HSV_Calibration_TrackBar', 0, 179, nothing)
        cv2.createTrackbar('S_low', 'HSV_Calibration_TrackBar', 0, 255, nothing)
        cv2.createTrackbar('V_low', 'HSV_Calibration_TrackBar', 0, 255, nothing)
        cv2.createTrackbar('H_high', 'HSV_Calibration_TrackBar', 0, 179, nothing)
        cv2.createTrackbar('S_high', 'HSV_Calibration_TrackBar', 0, 255, nothing)
        cv2.createTrackbar('V_high', 'HSV_Calibration_TrackBar', 0, 255, nothing)
        cv2.setTrackbarPos('H_low', 'HSV_Calibration_TrackBar', self.hsv_low.h)
        cv2.setTrackbarPos('S_low', 'HSV_Calibration_TrackBar', self.hsv_low.s)
        cv2.setTrackbarPos('V_low', 'HSV_Calibration_TrackBar', self.hsv_low.v)
        cv2.setTrackbarPos('H_high', 'HSV_Calibration_TrackBar', self.hsv_high.h)
        cv2.setTrackbarPos('S_high', 'HSV_Calibration_TrackBar', self.hsv_high.s)
        cv2.setTrackbarPos('V_high', 'HSV_Calibration_TrackBar', self.hsv_high.v)

        self.initialized = True
    
    def calibrate_ball_hsv_filter(self, image):
        self.hsv_low.h = cv2.getTrackbarPos('H_low','HSV_Calibration_TrackBar')
        self.hsv_low.s = cv2.getTrackbarPos('S_low','HSV_Calibration_TrackBar')
        self.hsv_low.v = cv2.getTrackbarPos('V_low','HSV_Calibration_TrackBar')
        self.hsv_high.h = cv2.getTrackbarPos('H_high','HSV_Calibration_TrackBar')
        self.hsv_high.s = cv2.getTrackbarPos('S_high','HSV_Calibration_TrackBar')
        self.hsv_high.v = cv2.getTrackbarPos('V_high','HSV_Calibration_TrackBar')

        #self.get_logger().info(f"HSV low: {self.hsv_low.h}-{self.hsv_low.s}-{self.hsv_low.v} /HSV high: {self.hsv_high.h}-{self.hsv_high.s}-{self.hsv_high.v}")

        image = cv2.flip( image, 1 )
        image = self.create_ball_mask(image)
        cv2.imshow("Calibrated", image)
        cv2.waitKey(1)
        
    def create_ball_mask(self, image):
        blur = cv2.blur(image,(3,3))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([self.hsv_low.h, self.hsv_low.s, self.hsv_low.v]),np.array([self.hsv_high.h, self.hsv_high.s, self.hsv_high.v]))
        return mask
	
    def publish_msgs(self, center):
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

