"""Dummy-state driver for testing mppi_ros_node end to end.

What this does:
  - Publishes a fake nav_msgs/Odometry on BASE_STATE_TOPIC at 50 Hz.
    The base traces a small circle so MPPI sees changing state.
  - Publishes a fake JointState on ARM_STATE_TOPIC at 50 Hz with the joint
    names the node expects (ARM_JOINT_NAMES) and zero positions/velocities.
  - Subscribes to PLAN_TOPIC and prints a one-line summary of each plan
    that arrives, so you can confirm the planner is running and producing
    sensible output.

Run alongside the planner:
  Terminal 1:  python3 mppi_ros_node.py
  Terminal 2:  python3 mppi_ros_test_driver.py
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

# Single source of truth for topic/joint names — pulled from the node itself
from mppi_ros_node import (
    ARM_STATE_TOPIC,
    ARM_JOINT_NAMES,
    BASE_STATE_TOPIC,
    PLAN_TOPIC,
)


class FakeStateDriver(Node):
    def __init__(self):
        super().__init__("mppi_test_driver")

        self.base_pub = self.create_publisher(
            Odometry, BASE_STATE_TOPIC, 10
        )
        self.arm_pub = self.create_publisher(
            JointState, ARM_STATE_TOPIC, 10
        )
        self.create_subscription(
            JointTrajectory, PLAN_TOPIC, self._plan_cb, 10
        )

        self.create_timer(0.02, self._publish_state)  # 50 Hz

        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.plan_count = 0
        self.get_logger().info(
            f"Driver up. Publishing state on:\n"
            f"  base: {BASE_STATE_TOPIC}\n"
            f"  arm:  {ARM_STATE_TOPIC}\n"
            f"Listening for plans on: {PLAN_TOPIC}\n"
            f"Arm joint names: {ARM_JOINT_NAMES}"
        )

    def _publish_state(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0

        # Fake base: 0.5 m radius circle in the xy-plane, 0.2 rad/s.
        # heading is fixed at 0, so body frame == world frame and we can
        # write the world-frame velocity straight into twist.linear (which
        # is supposed to be body-frame for Odometry, but they coincide
        # when yaw=0).
        r = 0.5
        omega = 0.2
        cos_, sin_ = math.cos(omega * t), math.sin(omega * t)

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = r * sin_
        odom.pose.pose.position.y = r * cos_
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = r * omega * cos_
        odom.twist.twist.linear.y = -r * omega * sin_
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0
        self.base_pub.publish(odom)

        # Fake arm: 6 joints, all at zero
        arm = JointState()
        arm.header.stamp = self.get_clock().now().to_msg()
        arm.name = list(ARM_JOINT_NAMES)
        arm.position = [0.0] * len(ARM_JOINT_NAMES)
        arm.velocity = [0.0] * len(ARM_JOINT_NAMES)
        self.arm_pub.publish(arm)

    def _plan_cb(self, msg: JointTrajectory):
        self.plan_count += 1

        # End-to-end latency: now() − plan.header.stamp. The planner sets
        # header.stamp to the timestamp of the Odometry that the plan was
        # computed from, so this covers state→subscribe→plan→publish→here.
        latency_ms = (
            self.get_clock().now() - Time.from_msg(msg.header.stamp)
        ).nanoseconds * 1e-6

        # Print every plan for the first 3, then 1 in 30 after that
        if self.plan_count > 3 and self.plan_count % 30 != 0:
            return
        first = msg.points[0].positions if msg.points else []
        n = len(msg.points)
        base_first = ", ".join(f"{x:+.3f}" for x in first[:3])
        arm_first = ", ".join(f"{x:+.3f}" for x in first[3:9])
        grip_first = first[9] if len(first) > 9 else float("nan")
        self.get_logger().info(
            f"plan #{self.plan_count}: {n} pts | "
            f"latency={latency_ms:6.1f}ms | "
            f"u0 base=[{base_first}] arm=[{arm_first}] grip={grip_first:+.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FakeStateDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
