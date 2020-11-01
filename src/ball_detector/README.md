# BallDetector package
Package that can detect ball on image.

# Status:
In progress / developing

# Prerequisters
ROS2/foxy + opencv

# Content:
ball_detector - main computation of ball detections

## usage:

ros2 run ball_detector ball_detector

## additional params:

doCalibration       - bool - decides if first there will be a calibration of HSV filers

doTracking          - bool - decides tracking of ball is done

showImage           - bool - decides if image should be shown

## input msgs:

/image : Image

## output msgs:

in progress

## TODO:

1) outputing messages (custom msg possible with Point and Radius)

2) stabilizing hough circle detections

3) computing distance to ball


