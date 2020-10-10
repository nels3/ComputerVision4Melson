from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='single_camera',
            namespace='vision/left',
            executable='camera_publisher',
            name='camera_left',
            parameters=[
                {"device": 2,
                "width":640,
                "height":480}
            ]
        ),Node(
            package='single_camera',
            namespace='vision/right',
            executable='camera_publisher',
            name='camera_right',
            parameters=[
                {"device": 0,
                "width":640,
                "height":480}
            ]
        )
    ])

