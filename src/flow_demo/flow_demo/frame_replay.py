#!/usr/bin/env python3
import os
import glob
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from PIL import Image as PILImage


class FrameReplay(Node):
    def __init__(self):
        super().__init__("frame_replay")

        # Parameters
        self.declare_parameter("frames_dir", "/workspaces/flow-ros2/frames")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("fps", 2.0)
        self.declare_parameter("loop", True)

        self.frames_dir = self.get_parameter("frames_dir").get_parameter_value().string_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)

        # Load frames
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        self.frame_paths = []
        for p in patterns:
            self.frame_paths += sorted(glob.glob(os.path.join(self.frames_dir, p)))

        if len(self.frame_paths) == 0:
            self.get_logger().error(f"No frames found in {self.frames_dir} (expected png/jpg).")
        else:
            self.get_logger().info(f"Found {len(self.frame_paths)} frames in {self.frames_dir}")

        # QoS for sensor streams
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # IMPORTANT: store publisher on self so it doesn't get GC'd
        self.pub = self.create_publisher(Image, self.topic, qos)

        self.idx = 0
        period = 1.0 / max(self.fps, 1e-6)

        # IMPORTANT: store timer on self
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Publishing to: {self.topic} at {self.fps:.2f} FPS | loop={self.loop}")

    def tick(self):
        if len(self.frame_paths) == 0:
            return

        if self.idx >= len(self.frame_paths):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("Done (loop=false). Shutting down.")
                rclpy.shutdown()
                return

        path = self.frame_paths[self.idx]

        try:
            pil = PILImage.open(path).convert("RGB")  # force 3-channel RGB
            w, h = pil.size
            data = pil.tobytes()  # row-major RGBRGB...
        except Exception as e:
            self.get_logger().error(f"Failed reading {path}: {e}")
            self.idx += 1
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = data

        self.pub.publish(msg)

        # log occasionally
        if self.idx == 0 or (self.idx + 1) % 10 == 0:
            self.get_logger().info(f"Published frame {self.idx+1}/{len(self.frame_paths)}: {os.path.basename(path)}")

        self.idx += 1


def main():
    rclpy.init()
    node = FrameReplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()