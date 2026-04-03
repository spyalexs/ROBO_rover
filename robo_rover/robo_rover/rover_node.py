#!/usr/bin/env python3
"""
ROS2 ArduPilot Rover Node
Combines steering/throttle control, IMU publishing, and odometry publishing
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from pymavlink import mavutil
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class ArduPilotRoverNode(Node):
    def __init__(self):
        super().__init__('rover_node')

        # Parameters
        self.declare_parameter('connection_string', '/dev/ttyACM1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('imu_frequency', 20.0)
        self.declare_parameter('odom_frequency', 20.0)
        self.declare_parameter('cmd_timeout', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # Get parameters
        self.connection_string = self.get_parameter('connection_string').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.control_freq = float(self.get_parameter('control_frequency').value)
        self.imu_freq = float(self.get_parameter('imu_frequency').value)
        self.odom_freq = float(self.get_parameter('odom_frequency').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # Control variables
        self.default_throttle = 0.0
        self.default_steering = 0.0
        self.current_throttle = self.default_throttle
        self.current_steering = self.default_steering
        self.last_cmd_time = time.time()
        self.last_cmd_linear = 0.0

        # Connection variables
        self.master = None
        self.connected = False
        self.armed = False

        # Cached MAVLink messages
        self.latest_scaled_imu = None
        self.latest_attitude = None
        self.latest_vfr_hud = None
        self.latest_heartbeat = None

        # Dead-reckoned odom state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.last_odom_update_time = None
        self.last_yaw_ros = 0.0
        self.last_groundspeed = 0.0

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers
        self.gyro_pub = self.create_publisher(Vector3, 'imu/gyro', sensor_qos)
        self.accel_pub = self.create_publisher(Vector3, 'imu/accel', sensor_qos)
        self.armed_pub = self.create_publisher(Bool, 'rover/armed', control_qos)
        self.odom_pub = self.create_publisher(Odometry, 'odom', control_qos)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, control_qos
        )

        # Timers
        self.control_timer = self.create_timer(
            1.0 / self.control_freq, self.control_loop
        )
        self.imu_timer = self.create_timer(
            1.0 / self.imu_freq, self.imu_loop
        )
        self.odom_timer = self.create_timer(
            1.0 / self.odom_freq, self.odom_loop
        )
        self.status_timer = self.create_timer(1.0, self.status_loop)

        # Single MAVLink polling timer to avoid recv_match() conflicts
        self.mavlink_timer = self.create_timer(0.01, self.mavlink_poll_loop)

        # Initialize connection
        self.get_logger().info('Initializing ArduPilot Rover Node...')
        self.connect_to_rover()

    def connect_to_rover(self):
        """Connect to the rover via MAVLink"""
        try:
            self.get_logger().info(f'Connecting to rover on {self.connection_string}...')

            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud_rate,
                timeout=10
            )

            # Wait for heartbeat
            self.get_logger().info('Waiting for heartbeat...')
            heartbeat = self.master.wait_heartbeat(timeout=10)

            if heartbeat is None:
                self.get_logger().error('No heartbeat received')
                return False

            self.get_logger().info(
                f'Connected to system {self.master.target_system} '
                f'component {self.master.target_component}'
            )

            self.connected = True

            # Set mode to ACRO
            if self.set_mode('ACRO'):
                time.sleep(2)
                self.arm_rover()
                self.request_imu_data()
                self.request_odom_data()

            return True

        except Exception as e:
            self.get_logger().error(f'Connection failed: {e}')
            return False

    def set_mode(self, mode_name):
        """Set the rover flight mode"""
        if not self.connected:
            return False

        mode_mapping = self.master.mode_mapping()
        if mode_name not in mode_mapping:
            self.get_logger().error(f'Mode {mode_name} not available')
            return False

        mode_id = mode_mapping[mode_name]

        self.get_logger().info(f'Setting mode to {mode_name}')

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        start_time = time.time()
        while time.time() - start_time < 5:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                current_mode = mavutil.mode_string_v10(msg)
                if current_mode == mode_name:
                    self.get_logger().info(f'Mode changed to {mode_name}')
                    return True
            time.sleep(0.1)

        self.get_logger().error(f'Failed to change mode to {mode_name}')
        return False

    def arm_rover(self):
        """Arm the rover"""
        if not self.connected:
            return False

        self.get_logger().info('Arming rover...')

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )

        start_time = time.time()
        while time.time() - start_time < 10:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    self.get_logger().info('Rover armed successfully')
                    self.armed = True
                    return True
            time.sleep(0.1)

        self.get_logger().error('Failed to arm rover')
        return False

    def disarm_rover(self):
        """Disarm the rover"""
        if not self.connected:
            return

        self.get_logger().info('Disarming rover...')

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0
        )

        self.armed = False

    def request_message_interval(self, msg_id, hz):
        """Helper to request MAVLink message rate"""
        interval_us = int(1000000 / hz)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0
        )

    def request_imu_data(self):
        """Request IMU messages from the autopilot"""
        if not self.connected:
            return

        try:
            self.request_message_interval(26, self.imu_freq)  # SCALED_IMU
            self.get_logger().info(f'Requested IMU data at {self.imu_freq} Hz')
        except Exception as e:
            self.get_logger().error(f'Failed to request IMU data: {e}')

    def request_odom_data(self):
        """Request attitude + speed messages for dead-reckoned odometry"""
        if not self.connected:
            return

        try:
            self.request_message_interval(30, self.odom_freq)  # ATTITUDE
            self.request_message_interval(74, self.odom_freq)  # VFR_HUD
            self.get_logger().info(
                f'Requested odom-related MAVLink data at {self.odom_freq} Hz'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to request odom data: {e}')

    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands"""
        throttle_raw = msg.linear.x * -400
        offset = 80

        if throttle_raw >= 0:
            throttle_with_offset = throttle_raw + offset
        else:
            throttle_with_offset = throttle_raw - offset

        self.current_throttle = int(np.clip(throttle_with_offset, -300, 300))
        self.current_steering = int(np.clip(msg.angular.z * 500, -1000, 1000))
        self.last_cmd_linear = msg.linear.x
        self.last_cmd_time = time.time()

        self.get_logger().debug(
            f'Received cmd_vel: throttle={self.current_throttle}, '
            f'steering={self.current_steering}, linear={self.last_cmd_linear}'
        )

    def mavlink_poll_loop(self):
        """Single MAVLink polling loop to avoid multiple recv_match() consumers"""
        if not self.connected:
            return

        try:
            while True:
                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    break

                msg_type = msg.get_type()

                if msg_type == 'SCALED_IMU':
                    self.latest_scaled_imu = msg
                elif msg_type == 'ATTITUDE':
                    self.latest_attitude = msg
                elif msg_type == 'VFR_HUD':
                    self.latest_vfr_hud = msg
                elif msg_type == 'HEARTBEAT':
                    self.latest_heartbeat = msg

        except Exception as e:
            self.get_logger().error(f'MAVLink polling error: {repr(e)}')
            raise

    def control_loop(self):
        """Main control loop - sends commands at fixed rate"""
        if not self.connected or not self.armed:
            return

        if time.time() - self.last_cmd_time > self.cmd_timeout:
            throttle = int(self.default_throttle * 1000)
            steering = int(self.default_steering * 1000)
        else:
            throttle = self.current_throttle
            steering = self.current_steering

        try:
            self.master.mav.manual_control_send(
                self.master.target_system,
                0,
                steering,
                throttle,
                0,
                0
            )
        except Exception as e:
            self.get_logger().error(f'Failed to send control command: {e}')

    def imu_loop(self):
        """Publish cached IMU data"""
        if not self.connected or self.latest_scaled_imu is None:
            return

        self.publish_scaled_imu(self.latest_scaled_imu)

    def publish_scaled_imu(self, scaled_imu_msg):
        gyro_msg = Vector3()
        gyro_msg.x = scaled_imu_msg.xgyro / 1000.0
        gyro_msg.y = scaled_imu_msg.ygyro / 1000.0
        gyro_msg.z = scaled_imu_msg.zgyro / 1000.0
        self.gyro_pub.publish(gyro_msg)

        accel_msg = Vector3()
        accel_msg.x = (scaled_imu_msg.xacc / 1000.0) * 9.80665
        accel_msg.y = (scaled_imu_msg.yacc / 1000.0) * 9.80665
        accel_msg.z = (scaled_imu_msg.zacc / 1000.0) * 9.80665
        self.accel_pub.publish(accel_msg)

    def odom_loop(self):
        """Publish dead-reckoned odometry from ATTITUDE + VFR_HUD"""
        if not self.connected:
            return

        if self.latest_attitude is None:
            return

        try:
            now_sec = time.time()
            if self.last_odom_update_time is None:
                self.last_odom_update_time = now_sec
                return

            dt = now_sec - self.last_odom_update_time
            self.last_odom_update_time = now_sec

            if dt <= 0.0 or dt > 1.0:
                return

            # ATTITUDE yaw from autopilot
            # ATTITUDE yaw from autopilot
            yaw_raw = float(self.latest_attitude.yaw)

            yaw_ros_current = (math.pi / 2.0) - yaw_raw
            yaw_ros_current = math.atan2(math.sin(yaw_ros_current), math.cos(yaw_ros_current))

            yaw_ros_neg = -yaw_raw
            yaw_ros_neg = math.atan2(math.sin(yaw_ros_neg), math.cos(yaw_ros_neg))

            yaw_ros_identity = math.atan2(math.sin(yaw_raw), math.cos(yaw_raw))

            groundspeed_dbg = float(self.latest_vfr_hud.groundspeed)

            self.get_logger().info(
                f"yaw_raw={math.degrees(yaw_raw):7.2f} deg | "
                f"curr(pi/2-raw)={math.degrees(yaw_ros_current):7.2f} deg | "
                f"neg(-raw)={math.degrees(yaw_ros_neg):7.2f} deg | "
                f"id(raw)={math.degrees(yaw_ros_identity):7.2f} deg | "
                f"gs={groundspeed_dbg:5.2f}"
            )

            # Keep using your current conversion for now
            yaw_ros = yaw_ros_current

            if time.time() - self.last_cmd_time > self.cmd_timeout:
                groundspeed = 0.0
            else:
                groundspeed = float(self.last_cmd_linear)

            self.odom_x += groundspeed * math.cos(yaw_ros) * dt
            self.odom_y += groundspeed * math.sin(yaw_ros) * dt
            self.last_yaw_ros = yaw_ros
            self.last_groundspeed = groundspeed

            quat = Rotation.from_euler('xyz', [0.0, 0.0, yaw_ros]).as_quat()
            stamp = self.get_clock().now().to_msg()

            odom_msg = Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = self.odom_frame
            odom_msg.child_frame_id = self.base_frame

            odom_msg.pose.pose.position.x = self.odom_x
            odom_msg.pose.pose.position.y = self.odom_y
            odom_msg.pose.pose.position.z = 0.0

            odom_msg.pose.pose.orientation.x = float(quat[0])
            odom_msg.pose.pose.orientation.y = float(quat[1])
            odom_msg.pose.pose.orientation.z = float(quat[2])
            odom_msg.pose.pose.orientation.w = float(quat[3])

            odom_msg.twist.twist.linear.x = groundspeed
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0

            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0

            # Some pymavlink ATTITUDE objects may not expose yawspeed reliably
            yaw_rate = getattr(self.latest_attitude, 'yawspeed', 0.0)
            odom_msg.twist.twist.angular.z = -float(yaw_rate)

            odom_msg.pose.covariance = [
                0.05, 0.0,  0.0,  0.0,  0.0,  0.0,
                0.0,  0.05, 0.0,  0.0,  0.0,  0.0,
                0.0,  0.0,  9999.0, 0.0,  0.0,  0.0,
                0.0,  0.0,  0.0,  9999.0, 0.0,  0.0,
                0.0,  0.0,  0.0,  0.0,  9999.0, 0.0,
                0.0,  0.0,  0.0,  0.0,  0.0,  0.1
            ]

            odom_msg.twist.covariance = [
                0.1,  0.0,  0.0,  0.0,  0.0,  0.0,
                0.0,  0.1,  0.0,  0.0,  0.0,  0.0,
                0.0,  0.0,  9999.0, 0.0,  0.0,  0.0,
                0.0,  0.0,  0.0,  9999.0, 0.0,  0.0,
                0.0,  0.0,  0.0,  0.0,  9999.0, 0.0,
                0.0,  0.0,  0.0,  0.0,  0.0,  0.2
            ]

            self.odom_pub.publish(odom_msg)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame

            tf_msg.transform.translation.x = self.odom_x
            tf_msg.transform.translation.y = self.odom_y
            tf_msg.transform.translation.z = 0.0

            tf_msg.transform.rotation.x = float(quat[0])
            tf_msg.transform.rotation.y = float(quat[1])
            tf_msg.transform.rotation.z = float(quat[2])
            tf_msg.transform.rotation.w = float(quat[3])

            self.tf_broadcaster.sendTransform(tf_msg)

        except Exception as e:
            self.get_logger().error(f'odom_loop failed: {repr(e)}')
            raise

    def status_loop(self):
        """Publish status information"""
        armed_msg = Bool()
        armed_msg.data = self.armed
        self.armed_pub.publish(armed_msg)

        if self.connected and self.latest_heartbeat is not None:
            self.armed = bool(
                self.latest_heartbeat.base_mode &
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )

    def destroy_node(self):
        """Clean up when node is destroyed"""
        self.get_logger().info('Shutting down rover node...')

        if self.connected and self.armed:
            try:
                self.master.mav.manual_control_send(
                    self.master.target_system,
                    0, 0, 0, 0, 0
                )
                time.sleep(0.1)
                self.disarm_rover()
            except Exception:
                pass

        if self.master:
            self.master.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ArduPilotRoverNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()