# PersonRecognition package
Package that can recognize person on image based on resources that contains previously trained model and can search if person on image has t-shirt with KNR Logo (Students' Robotics Association).

# Status:
In progress / developing

# Prerequisters
ROS2/foxy + opencv

pip3 install imutils
pip3 install scikit-learn==0.23.2

Resources:
1) openface_nn4.small2.v1.t7
2) recognizer.pickle
3) le.pickle
4) embeddings.pickle
5) face_detection_model
6) .png file with logo  

## Content:
person_recognition - main computation of searching for KNR person on image
usage:

ros2 run person_recognition person_recognition

additional params:

showImage             - bool - decides if image will be shown, default: True
searchForLogo         - bool - decides if there will be doing searching logo on image, default: False
drawLogoSearchWindow  - bool - will draw logo search window on image, default: False
visualizeIteration    - bool - for debug - visualization of steps for logo search, default: False
onlyNecessaryTask     - bool - for production - speeding up calculation by not visualizating things, default: False

usage:
ros2 run person_recognition person_recognition --ros-args -p showImage:=False onlyNecessaryTask:=True

input:
/image : Image

output:
/Person/Face/Roi        : PersonFaceList    - list of Rectangles and name for each Face Detection
/Person/Face/Image      : Image             - image with drawn result
/Person/Face/Detection  : Bool              - gives information if main player is detected on image

