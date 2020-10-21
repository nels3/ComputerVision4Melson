# HandSignalDetector package
Package that can detect hand signals. It is based on DLIB library.

# Status:
In progress / developing

# Prerequisters
ROS2/foxy + opencv + dlib

pip3 install dlib

Resources:
1) resource/Hand_Detector.svm 

# Content:

resource/create_training_dataset.py - script that can create dataset with hand images

resource/train_hand_detector.py - script that can train detector based on hand images

hand_signal_detector - main computation of hand detections

hand_gesture_processor - processor of hand position from hand_signal_detector 

## usage:

ros2 run hand_signal_detector hand_signal_detector

## additional params:

showImage       - bool - decides if image will be shown, default: False

showThresholds  - bool - decides if threshold should be visualized, default: False

showAll         - bool - decides if all information should be drawn on image, default: False

## input msgs:

/image : Image

## output msgs:

/Hand/Position : Int8 - output result of hand detection 

0) no detection
1) left up
2) left down
3) right up
4) right down
5) left bigger
6) right bigger
7) middle normal
8) middle bigger

