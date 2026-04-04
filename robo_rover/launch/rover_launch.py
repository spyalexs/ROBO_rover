#!/usr/bin/env python3
"""
ROS2 Launch file for ROBO Rover
Launches the rover node with configurable parameters
"""

import os
from launch.actions import DeclareLaunchArgument, LogInfo
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('robo_rover')
    urdf_file = os.path.join(package_share_dir, 'urdf', 'simple.urdf')

    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content
        }],
    )

    # Declare launch arguments
    connection_string_arg = DeclareLaunchArgument(
        'connection_string',
        default_value='/dev/ttyACM1',
        description='MAVLink connection string (serial port or UDP/TCP)'
    )

    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial connection'
    )

    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='20.0',
        description='Control command frequency in Hz'
    )

    imu_frequency_arg = DeclareLaunchArgument(
        'imu_frequency',
        default_value='20.0',
        description='IMU data publishing frequency in Hz'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the rover node'
    )
    
    # Rover node
    rover_node = Node(
        executable='python3',
        arguments=['-m', 'robo_rover.rover_node'],
        name='rover_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        emulate_tty=True,
        parameters=[{
            'connection_string': LaunchConfiguration('connection_string'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'imu_frequency': LaunchConfiguration('imu_frequency'),
        }],
    )
    # Static TF node
    static_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_link_to_laser',
    arguments=[
        '--x', '-0.0251', 
        '--y', '0.0', 
        '--z', '0.1683', 
        '--yaw', '0', 
        '--pitch', '0', 
        '--roll', '0', 
        '--frame-id', 'base_link', 
        '--child-frame-id', 'laser'
    ]
    )
    # Log info about the launch
    log_info = LogInfo(
        msg=[
            'Launching Rover Node with:\n',
            '  Connection: ', LaunchConfiguration('connection_string'), '\n',
            '  Baud Rate: ', LaunchConfiguration('baud_rate'), '\n',
            '  Control Frequency: ', LaunchConfiguration('control_frequency'), ' Hz\n',
            '  IMU Frequency: ', LaunchConfiguration('imu_frequency'), ' Hz\n',
        ]
    )

    return LaunchDescription([
        connection_string_arg,
        baud_rate_arg,
        control_frequency_arg,
        imu_frequency_arg,
        namespace_arg,
        log_info,
        rover_node,
        static_tf_node,
        robot_state_publisher_node,
    ])