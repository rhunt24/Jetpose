import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import TimerAction
from launch_ros.actions import Node

def generateLaunchDescription():
    package_name = 'ros2pose'

    realsense_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('realsense-ros'), 'realsense2_camera', 'launch', 'rs_launch.py')]),
    )

return LaunchDescription([
    realsense_camera,
])