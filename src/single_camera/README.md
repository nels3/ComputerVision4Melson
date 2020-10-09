# Single camera package

## Prerequisters:
ROS2: foxy (tested)
Installing: https://index.ros.org/p/cv_bridge/github-ros-perception-vision_opencv/

## Content

### camera_publisher executive
Can publish camera image from single camera to topic /image

usage:

ros2 run single_camera camera_publisher

additional params:

device                  - int - id of camera, default: 0
width                   - int - destiny image width, default: 640
height                  - int - destiny image height, default: 480
leave_original_image    - int - 1 means not setting width and height for image, 0 setting, default: 0

usage:
ros2 run single_camera camera_publisher --ros-args -p device:=0 -p leave_original_image:=1


### camera_subscriber executive

Can subscribe camera image from topic /image

usage:

ros2 run single_camera camera_subscriber




