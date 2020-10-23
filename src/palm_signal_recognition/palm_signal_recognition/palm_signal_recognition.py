import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int8
import numpy as np
import cv2
import os
import imutils

doCalibration = True

# Inital value for palm color threshold
h_low = 0
s_low = 63
v_low = 63
h_high = 179
s_high = 255
v_high = 255

class MyNode(Node):
    initialized:bool
    Roi = None
    hsv_low = None
    hsv_high = None
	
    class Rectangle():
        x:int
        y:int
        w:int
        h:int
        def __init__(self):
            self.x = 0
            self.y = 0
            self.w = 0
            self.h = 0
            pass
			
    class HSV_threshold():
        h:int
        s:int
        v:int
        def __init__(self, h, s, v):
            self.h = h
            self.s = s
            self.v = v
    
    def __init__(self):
        super().__init__('PersonRecognition')
        
        self.declare_parameter("do_calibration", doCalibration)
        do_calibration = self.get_parameter("do_calibration").value
				
        self.initialized = False
        
        self.Roi = self.Rectangle()
        self.hsv_low = self.HSV_threshold(h_low, s_low, v_low)
        self.hsv_high = self.HSV_threshold(h_high, s_high, v_high)
        
        if not do_calibration:
            self.image_sub = self.create_subscription(Image, 'image', self.listener_callback, 10)
            self.fingers_number_pub = self.create_publisher(Int8, 'person/palm/number', 10)
            self.image_pub = self.create_publisher(Image, 'person/palm/image', 10)
        else:
            self.image_sub = self.create_subscription(Image, 'image', self.calibration_listener_callback, 10)
			
        self.cv_bridge = CvBridge()
        
		
    def listener_callback(self, msg):
        """ Callback for image subscriber """
        bridge = CvBridge()

        image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        if not self.initialized:
            self.init_palm_signal_recognition(image)

        image, fingers_number = self.palm_signal_recognition(image)
        # self.publish_msgs(image, fingers_number)
         
        #cv2.imshow("Output image", image)
        cv2.waitKey(1)
		
    def calibration_listener_callback(self, msg):
        """ Callback for image subscriber """
        bridge = CvBridge()

        image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        if not self.initialized:
            self.init_calibration()

        self.calibrate_palm_hsv_filter(image)
        cv2.waitKey(1)
		
		
    def init_palm_signal_recognition(self, image):
        height_im, width_im, channels_im = image.shape
        self.Roi.x = int(0)
        self.Roi.w = int(width_im / 3)
        self.Roi.h = int(height_im / 2)
        self.Roi.y = int(height_im / 2)
		
        self.initialized = True
    
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
		
	
    def calibrate_palm_hsv_filter(self, image):
        self.hsv_low.h = cv2.getTrackbarPos('H_low','HSV_Calibration_TrackBar')
        self.hsv_low.s = cv2.getTrackbarPos('S_low','HSV_Calibration_TrackBar')
        self.hsv_low.v = cv2.getTrackbarPos('V_low','HSV_Calibration_TrackBar')
        self.hsv_high.h = cv2.getTrackbarPos('H_high','HSV_Calibration_TrackBar')
        self.hsv_high.s = cv2.getTrackbarPos('S_high','HSV_Calibration_TrackBar')
        self.hsv_high.v = cv2.getTrackbarPos('V_high','HSV_Calibration_TrackBar')

        self.get_logger().info(f"HSV low: {self.hsv_low.h}-{self.hsv_low.s}-{self.hsv_low.v} /HSV high: {self.hsv_high.h}-{self.hsv_high.s}-{self.hsv_high.v}")

        image = cv2.flip( image, 1 )
        image = self.create_palm_mask(image)
        cv2.imshow("Calibrated", image)
        cv2.waitKey(100)
		
    def create_palm_mask(self, image):
        blur = cv2.blur(image,(3,3))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        palm_mask = cv2.inRange(hsv, np.array([self.hsv_low.h, self.hsv_low.s, self.hsv_low.v]),np.array([self.hsv_high.h, self.hsv_high.s, self.hsv_high.v]))
        #cv2.imshow("Mask", palm_mask)
        return palm_mask

    def preprocess_image(self, image):
        palm_mask = self.create_palm_mask(image)

        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11), np.uint8)
        kernel_ellipse_5_x_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        kernel_ellipse_8_x_8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        kernel_ellipse_13_x_13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,12))

        #Perform morphological transformations 
        image = cv2.dilate(palm_mask, kernel_ellipse_5_x_5, iterations = 1)
        image = cv2.erode(image, kernel_square, iterations = 1)    
        image = cv2.dilate(image, kernel_ellipse_5_x_5, iterations = 1)
        image = cv2.medianBlur(image,5)
        image = cv2.dilate(image, kernel_ellipse_8_x_8,iterations = 1)
        image = cv2.erode(image, kernel_square, iterations = 1)    
        image = cv2.dilate(image, kernel_ellipse_13_x_13, iterations = 1)
        image = cv2.medianBlur(image, 5)
        ret,thresh = cv2.threshold(image, 127, 255, 0)
        #cv2.imshow("Morphological", thresh)

        return thresh
    
    def compute_distance(self, point, massCenter):
        return np.sqrt(np.power(point[0]-massCenter[0],2)+np.power(point[1]-massCenter[1],2))
        
    def find_fingers(self, image, original):
        fingers_number = 0
            
        #Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
            
	    #Find Max contour area
        max_area = 100
        palm_contour_index = 0	
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                palm_contour_index = i  
                
	    #Largest area contour 	
        if len(contours) == 0:
            return image, -1
        countour = contours[palm_contour_index]

        #Find convex hull
        hull = cv2.convexHull(countour)
        
        #Find convex defects
        hull2 = cv2.convexHull(countour, returnPoints = False)
        defects = cv2.convexityDefects(countour, hull2)
        
	    #Find moments of the largest contour
        moments = cv2.moments(countour)
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) 
            cy = int(moments['m01']/moments['m00']) 
        centerMass=(cx,cy)  
        cv2.circle(original, centerMass, 10, [0, 0, 255], 3)
        
        #Get defect points and draw them in the original image
        FarDefectsList = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(countour[s][0])
            end = tuple(countour[e][0])
            far = tuple(countour[f][0])
            FarDefectsList.append(far)
            
            cv2.line(image, start, end, [0, 255, 0], 1)
            cv2.circle(image, far, 10, [100, 255, 255], 3)
        
        distanceToCenter = []
        centerMassNp = np.array(centerMass)
        
        for i in range(0, len(FarDefectsList)):
            x =  np.array(FarDefectsList[i])
            distanceToCenter.append(self.compute_distance(x, centerMassNp))
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDistanceToCenter = sorted(distanceToCenter)
        minDefectDistance = np.mean(sortedDistanceToCenter[0:2]) + 140
        cv2.circle(original, centerMass, int(minDefectDistance),[0,0,255], 3)
     
        #Get candidates for points
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 40) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 40):
                if hull[i][0][1] < 600:
                    finger.append(hull[i][0])
                
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger, key=lambda x: x[1])  
        fingers = finger[0:5]
		
        #Calculate distance of each finger tip to the center mass
        for i in range(0,len(fingers)):
            distance = self.compute_distance(fingers[i], centerMassNp)
            if distance >  minDefectDistance:
                cv2.circle(original,(fingers[i][0], fingers[i][1]),10,[255,0,0], 3)
                fingers_number = fingers_number +1
            else:
                #print(f"{distance} vs {minDefectDistance}")
                cv2.circle(original,(fingers[i][0], fingers[i][1]),10,[255,0,255], 2)
             
                     
        #Print number of pointed fingers
        cv2.putText(image,str(fingers_number),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        cv2.putText(original,str(fingers_number),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
            
        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(countour)
        img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.drawContours(image,[hull],-1,(255,255,255),2)
        
        cv2.imshow("Output", image)
        cv2.imshow("Original", original)
        
        return image, fingers_number
        
    def palm_signal_recognition(self, image):
        image = cv2.flip( image, 1 )
        image_Roi = image[self.Roi.y:self.Roi.y+self.Roi.h, self.Roi.x:self.Roi.x+self.Roi.w]
        image_Roi = cv2.resize(image_Roi, (600,800) )
        image = cv2.rectangle(image, (self.Roi.x, self.Roi.y), (self.Roi.x + self.Roi.w, self.Roi.y + self.Roi.h), (0,255,0), 2)
        cv2.imshow("ROI", image)
        
        image_preprocessed = self.preprocess_image(image_Roi)
        
        output, fingers_number = self.find_fingers(image_preprocessed, image_Roi)
        
        return output, fingers_number
        
	
    def publish_msgs(self, image, fingers_number):
        """ Publishing msgs """
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image))
        self.fingers_number_pub.publish(fingers_number)
            
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

