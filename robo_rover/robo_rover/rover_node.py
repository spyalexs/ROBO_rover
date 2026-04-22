#!/usr/bin/env python3
"""
ROS2 ArduPilot Rover Node
Combines steering/throttle control, IMU publishing, and odometry publishing
"""

import math
import time
from pymavlink import mavutil
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from pymavlink import mavutil
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Vector3
import os
import time
from math import exp
from types import SimpleNamespace

import numpy as np
import rclpy
import yaml
from ament_index_python import get_package_share_directory
from geometry_msgs.msg import TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from pymavlink import mavutil
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

STEER_SERVO = 4
DRIVE_SERVO = 3
PWM_MAX = 2000
NEUTRAL_PWM = 1500
PWM_MIN = 1000

OL_MODEL_SUBPATH = "resource/ol_data.yaml"


class ArduPilotRoverNode(Node):
    def __init__(self):
        super().__init__('rover_node')

        # Parameters
        self.declare_parameter('connection_string', '/dev/ttyACM1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('imu_frequency', 20.0)
        self.declare_parameter('manual_mode', False)
        self.declare_parameter('ol_rate_mapping', True)
        self.declare_parameter('odom_frequency', 20.0)
        self.declare_parameter('cmd_timeout', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gyro_cal_duration', 3.0)
        self.declare_parameter('imu_stale_timeout', 0.20)
        self.declare_parameter('debug_gyro_yaw', False)
        self.declare_parameter('cmd_vel_scale', 1.0)

        # Get parameters
        self.connection_string = self.get_parameter('connection_string').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.control_freq = float(self.get_parameter('control_frequency').value)
        self.imu_freq = float(self.get_parameter('imu_frequency').value)
        self.is_manual = self.get_parameter('manual_mode').value
        self.manaual_rate_mapping = self.get_parameter('ol_rate_mapping').value
        self.odom_freq = float(self.get_parameter('odom_frequency').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.gyro_cal_duration = float(self.get_parameter('gyro_cal_duration').value)
        self.imu_stale_timeout = float(self.get_parameter('imu_stale_timeout').value)
        self.debug_gyro_yaw = bool(self.get_parameter('debug_gyro_yaw').value)
        self.cmd_vel_scale = float(self.get_parameter('cmd_vel_scale').value)

        # Control variables
        self.default_throttle = 0.0
        self.default_steering = 0.0
        self.current_throttle = self.default_throttle
        self.current_steering = self.default_steering
        self.last_cmd_time = time.time()
        self.last_cmd_time_ros = self.get_clock().now()
        self.last_nonzero_cmd_time = self.get_clock().now()
        self.last_cmd_linear = 0.0

        # Connection variables
        self.master = None
        self.connected = False
        self.armed = False

        # Cached MAVLink messages
        self.latest_scaled_imu = None
        self.latest_scaled_imu_ros_time = None
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

        # Gyro-integrated yaw state
        self.gyro_bias_z = 0.0         # rad/s
        self.gyro_bias_sum = 0.0
        self.gyro_bias_count = 0
        self.gyro_bias_ready = False
        self.gyro_cal_start_time = None
        self.yaw_initialized = False

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
        self.ol_rate_pub = self.create_publisher(Twist, 'ol_rates', sensor_qos)
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
        self.param_timer = self.create_timer(1.0, self.param_cb)
        self.mavlink_timer = self.create_timer(0.01, self.mavlink_poll_loop)

        self.current_ol_velocity = 0.0
        self.ol_stamp = self.get_ros_time_as_double()

        self.ol_model_loaded = False
        self.load_ol_model()

        # Initialize connection
        self.get_logger().info('Initializing ArduPilot Rover Node...')
        self.connect_to_rover()

    @staticmethod
    def wrap_pi(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def load_ol_model(self):
        # get the robo rover share directory
        robo_share_dir = get_package_share_directory("robo_rover")
        ol_path = os.path.join(robo_share_dir, OL_MODEL_SUBPATH)

        with open(ol_path, "r") as file:
            try:
                model_raw = yaml.safe_load(file)
            except Exception:
                self.get_logger().error("Failed to open ol model!")
                return

        self.ol_model = SimpleNamespace()
        self.ol_model.velocity = SimpleNamespace()
        self.ol_model.steering = SimpleNamespace()
        self.ol_model.time_constant = model_raw["time_constant"]
        self.ol_model.velocity.ol_velocities = np.array(
            model_raw["velocity"]["steady_state_velocity"]
        )
        self.ol_model.velocity.pwms = np.array(model_raw["velocity"]["pwm_values"])
        self.ol_model.steering.ol_radius = np.array(
            model_raw["angular"]["turn_radius_values"]
        )
        self.ol_model.steering.pwms = np.array(model_raw["angular"]["pwm_values"])

        self.ol_model_loaded = True

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

            desired_mode = "ACRO"
            if self.is_manual:
                desired_mode = "MANUAL"
                self.get_logger().warn("Must move steering servo into servo slot 4!")
            else:
                self.get_logger().warn("Must move steering servo into servo slot 2!")

            if self.set_mode(desired_mode):
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

        # Wait for mode change confirmation
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

        # Wait for arming confirmation
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
        if self.is_manual:
            # Safety first: manual mode expects raw PWM values.
            if (msg.linear.x < PWM_MIN) or (msg.linear.x > PWM_MAX):
                self.current_throttle = NEUTRAL_PWM
            else:
                self.current_throttle = msg.linear.x

            if (msg.angular.z < PWM_MIN) or (msg.angular.z > PWM_MAX):
                self.current_steering = NEUTRAL_PWM
            else:
                self.current_steering = msg.angular.z

            self.last_cmd_linear = 0.0
            self.last_cmd_time = time.time()
            self.last_cmd_time_ros = self.get_clock().now()
            return

        # adds offset to throttle to make it act more linear
        throttle_raw = msg.linear.x * -400
        offset = 80

        if throttle_raw >= 0:
            throttle_with_offset = throttle_raw + offset
        else:
            throttle_with_offset = throttle_raw - offset

        self.current_throttle = int(np.clip(throttle_with_offset, -300, 300))
        self.current_steering = int(np.clip(msg.angular.z * 500, -1000, 1000))
        self.last_cmd_linear = float(msg.linear.x)
        self.last_cmd_time_ros = self.get_clock().now()
        if abs(msg.linear.x) > 0.01:
            self.last_nonzero_cmd_time = self.get_clock().now()

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
                    self.latest_scaled_imu_ros_time = self.get_clock().now()
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
            self.rate_mapping_cb()
            return

        if self.is_manual:
            if time.time() - self.last_cmd_time > self.cmd_timeout:
                throttle = NEUTRAL_PWM
                steering = NEUTRAL_PWM
            else:
                throttle = self.current_throttle
                steering = self.current_steering

            self.set_servo_pwm(STEER_SERVO, steering)
            self.set_servo_pwm(DRIVE_SERVO, throttle)
            self.rate_mapping_cb()
            return

        now = self.get_clock().now()
        cmd_age = (now - self.last_cmd_time_ros).nanoseconds / 1e9

        if cmd_age > self.cmd_timeout:
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

    def rate_mapping_cb(self):
        if self.manaual_rate_mapping and self.is_manual and self.ol_model_loaded:
            # publish a mapping of the inputs to an open loop rate
            steady_state_vel = self.get_velocity_ol_steady_state()

            # treat this as a first order system
            current_time = self.get_ros_time_as_double()
            gap_closure = exp(
                -1 * (current_time - self.ol_stamp) / self.ol_model.time_constant
            )
            self.current_ol_velocity = (
                (self.current_ol_velocity - steady_state_vel) * gap_closure
                + steady_state_vel
            )

            msg = Twist()
            msg.linear.x = self.current_ol_velocity
            msg.angular.z = float(self.current_ol_velocity) / float(
                self.get_turn_radius_ol()
            )
            self.ol_rate_pub.publish(msg)

            self.ol_stamp = current_time

    def get_velocity_ol_steady_state(self):
        greater_pwms = np.where(self.ol_model.velocity.pwms > self.current_throttle)[0]

        if len(greater_pwms) == 0:
            return self.ol_model.velocity.ol_velocities[0]

        h_pwm = greater_pwms[0]
        l_pwm = greater_pwms[0] - 1
        if h_pwm == 0:
            return self.ol_model.velocity.ol_velocities[0]

        # linearly interpolate
        m = (
            self.ol_model.velocity.ol_velocities[h_pwm]
            - self.ol_model.velocity.ol_velocities[l_pwm]
        ) / (
            self.ol_model.velocity.pwms[h_pwm]
            - self.ol_model.velocity.pwms[l_pwm]
        )
        b = self.ol_model.velocity.ol_velocities[l_pwm]

        return b + m * (self.current_throttle - self.ol_model.velocity.pwms[l_pwm])

    def get_turn_radius_ol(self):
        greater_pwms = np.where(self.ol_model.steering.pwms > self.current_steering)[0]

        if len(greater_pwms) == 0:
            return self.ol_model.steering.ol_radius[0]

        h_pwm = greater_pwms[0]
        l_pwm = greater_pwms[0] - 1
        if h_pwm == 0:
            return self.ol_model.steering.ol_radius[0]

        # linearly interpolate
        m = (
            self.ol_model.steering.ol_radius[h_pwm]
            - self.ol_model.steering.ol_radius[l_pwm]
        ) / (
            self.ol_model.steering.pwms[h_pwm]
            - self.ol_model.steering.pwms[l_pwm]
        )
        b = self.ol_model.steering.ol_radius[l_pwm]

        return b + m * (self.current_steering - self.ol_model.steering.pwms[l_pwm])

    # def get_odom_groundspeed(self, now):
    #     if self.is_manual:
    #         if time.time() - self.last_cmd_time > self.cmd_timeout:
    #             return 0.0
    #         if self.manaual_rate_mapping and self.ol_model_loaded:
    #             # Guard against the un-initialised throttle value (0.0) which
    #             # sits outside the PWM table and would return a bogus 5.97 m/s.
    #             if not (PWM_MIN <= self.current_throttle <= PWM_MAX):
    #                 return 0.0
    #             # Use instantaneous steady-state lookup instead of the lagged
    #             # current_ol_velocity.  Forward PWM (<~1415) gives positive
    #             # velocity in ol_data.yaml, which matches the odom convention
    #             # (positive groundspeed = forward along the robot x-axis).
    #             return float(self.get_velocity_ol_steady_state())
    #         return 0.0

    #     cmd_age = (now - self.last_cmd_time_ros).nanoseconds / 1e9
    #     if cmd_age > self.cmd_timeout:
    #         return 0.0
    #     return float(self.last_cmd_linear)
    
    def odom_loop(self):
        """Publish dead-reckoned odometry using integrated SCALED_IMU z-gyro + commanded speed"""
        if not self.connected:
            return

        if self.latest_scaled_imu is None:
            return

        now = self.get_clock().now()

        # Initialize timers
        if self.last_odom_update_time is None:
            self.last_odom_update_time = now
            self.gyro_cal_start_time = now
            return

        dt = (now - self.last_odom_update_time).nanoseconds / 1e9
        self.last_odom_update_time = now

        if dt <= 0.0 or dt > 1.0:
            return

        # Reject very stale IMU data
        if self.latest_scaled_imu_ros_time is None:
            return

        imu_age = (now - self.latest_scaled_imu_ros_time).nanoseconds / 1e9
        if imu_age > self.imu_stale_timeout:
            self.get_logger().warn(
                f'SCALED_IMU stale ({imu_age:.3f}s old); skipping odom update once'
            )
            return

        # SCALED_IMU zgyro is in mrad/s -> rad/s
        wz_meas = float(self.latest_scaled_imu.zgyro) / 1000.0

        # -----------------------------
        # Startup gyro bias calibration
        # Keep robot still for a few seconds after launch
        # -----------------------------
        if not self.gyro_bias_ready:
            elapsed = (now - self.gyro_cal_start_time).nanoseconds / 1e9

            if elapsed < self.gyro_cal_duration:
                self.gyro_bias_sum += wz_meas
                self.gyro_bias_count += 1

                if not self.yaw_initialized:
                    self.last_yaw_ros = 0.0
                    self.yaw_initialized = True

                self.get_logger().info(
                    f'Calibrating gyro bias... keep robot still '
                    f'({elapsed:.1f}/{self.gyro_cal_duration:.1f}s)'
                )
                return

            if self.gyro_bias_count > 0:
                self.gyro_bias_z = self.gyro_bias_sum / self.gyro_bias_count
            else:
                self.gyro_bias_z = 0.0

            self.gyro_bias_ready = True
            self.last_yaw_ros = 0.0
            self.yaw_initialized = True

            self.get_logger().info(
                f'Gyro bias calibration done: '
                f'{self.gyro_bias_z:.6f} rad/s ({math.degrees(self.gyro_bias_z):.4f} deg/s)'
            )
            return

        # -----------------------------
        # Integrate gyro z to get yaw
        # -----------------------------
        wz_unbiased = wz_meas - self.gyro_bias_z

        # Based on your test:
        # right turn -> integrated yaw went positive
        # left turn  -> integrated yaw went negative
        # For ROS yaw, we want left positive and right negative, so flip sign here.
        delta_yaw = -(wz_unbiased * dt)

        yaw_old = self.last_yaw_ros
        yaw_mid = self.wrap_pi(yaw_old + 0.5 * delta_yaw)
        yaw_new = self.wrap_pi(yaw_old + delta_yaw)

        groundspeed = self.last_cmd_linear * self.cmd_vel_scale

        # Midpoint integration for Ackermann-like arcs
        self.odom_x += groundspeed * math.cos(yaw_mid) * dt
        self.odom_y += groundspeed * math.sin(yaw_mid) * dt

        self.last_yaw_ros = yaw_new
        self.last_groundspeed = groundspeed

        if self.debug_gyro_yaw:
            self.get_logger().info(
                f'yaw={math.degrees(yaw_new):7.2f} deg | '
                f'delta={math.degrees(delta_yaw):7.2f} deg | '
                f'wz={math.degrees(wz_unbiased):7.2f} deg/s | '
                f'cmd_v={groundspeed:5.2f}'
            )

        quat = Rotation.from_euler('xyz', [0.0, 0.0, yaw_new]).as_quat()
        stamp = now.to_msg()

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
        odom_msg.twist.twist.angular.z = delta_yaw / dt if dt > 0.0 else 0.0

        # Covariances: planar rover, approximate dead reckoning
        odom_msg.pose.covariance = [
            0.15, 0.0,  0.0,    0.0,    0.0,    0.0,
            0.0,  0.15, 0.0,    0.0,    0.0,    0.0,
            0.0,  0.0,  9999.0, 0.0,    0.0,    0.0,
            0.0,  0.0,  0.0,    9999.0, 0.0,    0.0,
            0.0,  0.0,  0.0,    0.0,    9999.0, 0.0,
            0.0,  0.0,  0.0,    0.0,    0.0,    0.4
        ]

        odom_msg.twist.covariance = [
            0.20, 0.0,  0.0,    0.0,    0.0,    0.0,
            0.0,  0.20, 0.0,    0.0,    0.0,    0.0,
            0.0,  0.0,  9999.0, 0.0,    0.0,    0.0,
            0.0,  0.0,  0.0,    9999.0, 0.0,    0.0,
            0.0,  0.0,  0.0,    0.0,    9999.0, 0.0,
            0.0,  0.0,  0.0,    0.0,    0.0,    0.3
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

    def param_cb(self):
        # callback for handling changed robo rover parameters
        if self.is_manual != self.get_parameter('manual_mode').value:
            # make sure these are neutral before changing modes
            self.set_servo_pwm(3, NEUTRAL_PWM)
            self.set_servo_pwm(4, NEUTRAL_PWM)

            self.is_manual = self.get_parameter('manual_mode').value
            self.last_cmd_linear = 0.0
            self.last_cmd_time = time.time()
            self.last_cmd_time_ros = self.get_clock().now()

            self.get_logger().warn(
                "If moving to manual mode, move the steer servo to 4. "
                "If moving to acro, move steer servo to 2!"
            )

            desired_mode = "MANUAL" if self.is_manual else "ACRO"
            self.set_mode(desired_mode)

        ol_mapping = self.get_parameter('ol_rate_mapping').value
        if ol_mapping != self.manaual_rate_mapping:
            self.manaual_rate_mapping = ol_mapping

            if ol_mapping:
                self.current_ol_velocity = self.get_velocity_ol_steady_state()
                self.ol_stamp = self.get_ros_time_as_double()

        if not (self.connection_string == self.get_parameter('connection_string').value):
            self.connection_string = self.get_parameter('connection_string').value

            #re init the connection
            self.connect_to_rover()

    def get_ros_time_as_double(self):
        # return the ros2 time as float
        now_sec, now_nsec = self.get_clock().now().seconds_nanoseconds()
        return now_sec + now_nsec * 1e-9

    def destroy_node(self):
        """Clean up when node is destroyed"""
        self.get_logger().info('Shutting down rover node...')

        if self.connected and self.armed:
            try:
                if self.is_manual:
                    self.set_servo_pwm(STEER_SERVO, NEUTRAL_PWM)
                    self.set_servo_pwm(DRIVE_SERVO, NEUTRAL_PWM)
                else:
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

    def set_servo_pwm(self, servo, pwm):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
            servo, pwm, 0, 0, 0, 0, 0
        )


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