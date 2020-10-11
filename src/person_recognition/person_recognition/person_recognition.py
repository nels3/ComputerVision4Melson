import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msg_and_srv.msg import PersonFaceList
from custom_msg_and_srv.msg import PersonFace
import numpy as np
import cv2
import os
import imutils
import pickle

# PATH TO RESOURCES
dir_path = "/home/nels/Desktop/ComputerVision4Melson/src/person_recognition/person_recognition"
detectorPath = "resources/face_detection_model"
embeddingModelPath = "resources/openface_nn4.small2.v1.t7"
recognizerPath = "resources/recognizer.pickle"
lePath = "resources/le.pickle"
min_confidence = 0.3

templatePath = "resources/logo_original.png"

#TODO: move to params
visualizeIteration = False
drawLogoSearchWindow = False
searchForLogo = True

class MyNode(Node):
    class Rectangle():
        x:int
        y:int
        w:int
        h:int
        found:bool
        confidence:int
        def __init__(self):
            self.x = 0
            self.y = 0
            self.w = 0
            self.h = 0
            self.found = False
            pass
    
    person_color_mapper = {
        "Kornelia" : (0, 255, 0),
        "Mateusz" : (255,0,0),
        "Other": (0, 0, 255)
    }

    def __init__(self):
        super().__init__('PersonRecognition')
        self.image_sub = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.person_face_list_pub = self.create_publisher(PersonFaceList, 'person/face/roi', 10)
        self.image_pub = self.create_publisher(Image, 'person/face/image', 10)

        self.cv_bridge = CvBridge()

        # Initialization of face recognition variables
        self.init_face_recognition()

        # Initialization of logo recodnition variables
        if searchForLogo:
            self.init_logo_finding_template()

    def listener_callback(self, msg):
        """ Callback for image subscriber """
        bridge = CvBridge()

        image = bridge.imgmsg_to_cv2(msg, 'bgr8')

        image, person_face_list = self.recognize_face(image)
        self.publish_msgs(image, person_face_list)
         
        cv2.imshow("Output image", image)
        cv2.waitKey(1)
		
    def init_face_recognition(self):
        self.get_logger().info("Starting loading face detection resources")

        # Loading face detector
        protoFullPath = os.path.sep.join([dir_path, detectorPath, "deploy.prototxt"])
        modelFullPath = os.path.sep.join([dir_path, detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoFullPath, modelFullPath)

        # Loading other necessary resources
        embedderFullPath = os.path.sep.join([dir_path, embeddingModelPath])
        recognizerFullPath = os.path.sep.join([dir_path, recognizerPath])
        leFullPath = os.path.sep.join([dir_path, lePath])
        self.embedder = cv2.dnn.readNetFromTorch(embedderFullPath)
        self.recognizer = pickle.loads(open(recognizerFullPath, "rb").read())
        self.le = pickle.loads(open(leFullPath, "rb").read())

        self.get_logger().info("Loaded face detection resources!")
	
    def init_logo_finding_template(self):
        # Load template for logo
        templateFullPath = os.path.sep.join([dir_path, templatePath])
        
        self.template = cv2.imread(templateFullPath)
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template = cv2.Canny(self.template, 50, 200)
        (self.template_height, self.template_width) = self.template.shape[:2]
        cv2.imshow("Logo template", self.template)
        cv2.waitKey(1000)
        self.get_logger().info("Loaded logo resources!")
	
    def compute_roi_for_logo_recognition(self, face: Rectangle, height: int, width: int) -> Rectangle:
        """Computing ROI for searching logo on image"""
        roi = self.Rectangle()
        roi.x = face.x - face.w
        roi.y = face.y + face.h
        roi.w = 3 * face.w
        roi.h = 2 * face.h
        
        # Sanity check if ROI is inside original image
        if roi.x < 0:
            roi.x = 0
        if roi.x + roi.w > width:
            roi.w =  width - roi.w
        if roi.y > height:
            roi.y = height
        if roi.y + roi.h > height:
            roi.h = height - roi.y

        return roi
	
    def find_logo(self, image, gray, face: Rectangle):
        """ Finding logo in image under face rectangle"""
        # Compute search window for logo
        height_im, width_im, channels_im = image.shape
        logo_roi = self.compute_roi_for_logo_recognition(face, height_im, width_im)
		   
        if drawLogoSearchWindow:
            cv2.rectangle(image, (logo_roi.x, logo_roi.y), (logo_roi.x+logo_roi.w, logo_roi.y+logo_roi.h), (255, 0, 0), 1)

        # Crop image for logo ROI
        grayLogoImg = gray[logo_roi.y:logo_roi.y+logo_roi.h, logo_roi.x:logo_roi.x+logo_roi.w]

        if grayLogoImg.shape[0] == 0 or grayLogoImg.shape[1] == 0:
            return None, image
	        
        foundLogo = None
        logo_global = None
        
        # Loop over the scales of the image to found best match for logo
        for scale in np.linspace(0.001, 1.1, 30)[::-1]:
            resizedGrayLogoImg = imutils.resize(grayLogoImg, width = int(grayLogoImg.shape[1] * scale))
            r = grayLogoImg.shape[1] / float(resizedGrayLogoImg.shape[1])

            # If the resized image is smaller than the template -> break
            if resizedGrayLogoImg.shape[0] < self.template_height or resizedGrayLogoImg.shape[1] < self.template_width:
	            break

            # Detect edges in the resized and match the template
            edged = cv2.Canny(resizedGrayLogoImg, 50, 200)
            result = cv2.matchTemplate(edged, self.template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # Visualization of iteration
            if visualizeIteration:
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + self.template_width, maxLoc[1] + self.template_height), (0, 0, 255), 2)

                cv2.imshow("Template searching", clone)
                cv2.waitKey(1000)

            # Saving of best match
            if foundLogo is None or maxVal > foundLogo[0]:
                foundLogo = (maxVal, maxLoc, r)

        # Show results om image
        if foundLogo is not None:
            logo = self.Rectangle()
            (_, maxLoc, r) = foundLogo
            (logo.x, logo.y) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (logo.w, logo.h) = (int(self.template_width * r), int(self.template_height * r))
            
            if foundLogo[2] > 0.4:
                color = (0,255,0)
            elif foundLogo[2] > 0.2:
                color = (0,255,255)
            else:
                return logo_global, image

            # Saving result in global reference system
            logo_global = self.Rectangle()
            logo_global.found = True
            logo_global.x = logo_roi.x+logo.x
            logo_global.y = logo_roi.y+logo.y
            logo_global.w = logo.w
            logo_global.h = logo.h 
            logo_global.confidence = foundLogo[2]
            
            cv2.rectangle(image, (logo_global.x, logo_global.y), (logo_global.x+logo_global.w, logo_global.y+logo_global.h), color, 2)
        return logo_global, image
    
    def pack_person_data_to_msg(self, name, face, logo):
        """ Packing data to PersonFace msg """
        person_face_elem = PersonFace()
        person_face_elem.x = face.x
        person_face_elem.y = face.y
        person_face_elem.w = face.y
        person_face_elem.h = face.h

        person_face_elem.name = name
               
        if logo is not None:
            person_face_elem.found_logo = logo.found
            if logo.found:
                person_face_elem.logo_x = logo.x
                person_face_elem.logo_y = logo.y
                person_face_elem.logo_w = logo.y
                person_face_elem.logo_h = logo.h
        else:
            person_face_elem.found_logo = False
            
        
        return person_face_elem
		 
    def recognize_face(self, image):
        """ Main computation of recognizing faces"""
        # Resizing input image
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # Constructing a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Searching for face on image with the use of Opencv
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
		
        person_face_list = []

        # Loop over the detections of faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")				
                face_img = image[startY:endY, startX:endX]

                (face_img_height, face_img_width) = face_img.shape[:2]

                # Checking if faces are large enough
                if face_img_height < 20 or face_img_width < 20:
	                continue

                faceBlob = cv2.dnn.blobFromImage(face_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j] 

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	                
                # Draw the bounding box of the face on image
                color = self.person_color_mapper[name]
                
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),color, 2)
                cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  

                face = self.Rectangle()
                face.x = int(startX)
                face.y = int(startY)
                face.w = int(endX - startX)
                face.h = int(endY - startY)
        
                # Searching for logo under person face
                if searchForLogo:
                    logo, image = self.find_logo(image, gray, face)
                else:
                    logo = None
                
                person_face_elem = self.pack_person_data_to_msg(name, face, logo)
                person_face_list.append(person_face_elem)
		
        return image, person_face_list
	
    def publish_msgs(self, image, person_face_list):
        """ Publishing msgs """
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image))

        person_face_list_msg = PersonFaceList()
        person_face_list_msg.list = person_face_list
        self.person_face_list_pub.publish(person_face_list_msg)
            
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

