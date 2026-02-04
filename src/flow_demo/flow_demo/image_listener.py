#!/usr/bin/env python3
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image

# Optional: OpenCV + cv_bridge
USE_CV = True
try:
    import cv2
    from cv_bridge import CvBridge
except Exception:
    USE_CV = False


class ImageListener(Node):
    def __init__(self):
        super().__init__("image_listener")

        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("out_dir", "/workspaces/flow-ros2/frames_rx")
        self.declare_parameter("save_every", 10)     # save every N frames
        self.declare_parameter("max_saves", 50)      # stop after this many saved frames

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.out_dir = self.get_parameter("out_dir").get_parameter_value().string_value
        self.save_every = int(self.get_parameter("save_every").value)
        self.max_saves = int(self.get_parameter("max_saves").value)

        os.makedirs(self.out_dir, exist_ok=True)

        # Match camera-like QoS (BEST_EFFORT)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(Image, self.topic, self.cb, qos)

        self.bridge = CvBridge() if USE_CV else None
        self.frame_idx = 0
        self.saved = 0
        self.t0 = time.time()

        self.get_logger().info(
            f"Subscribing to {self.topic} | saving to {self.out_dir} | "
            f"USE_CV={USE_CV} | save_every={self.save_every} | max_saves={self.max_saves}"
        )

    def cb(self, msg: Image):
        self.frame_idx += 1

        # Basic rate log
        if self.frame_idx % 30 == 0:
            dt = time.time() - self.t0
            hz = self.frame_idx / max(dt, 1e-6)
            self.get_logger().info(f"rx frames={self.frame_idx} (~{hz:.2f} Hz)")

        # Save occasionally
        if (self.frame_idx % self.save_every) != 0:
            return

        fname = os.path.join(self.out_dir, f"rx_{self.frame_idx:06d}.png")

        try:
            if USE_CV:
                # Convert ROS Image -> OpenCV BGR
                # Supports rgb8 and bgr8 commonly; cv_bridge handles encoding
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(fname, cv_img)
            else:
                # Fallback: just dump raw bytes (no decoding)
                # (You can later decode if encoding is known.)
                raw_path = fname.replace(".png", f".{msg.encoding}.raw")
                with open(raw_path, "wb") as f:
                    f.write(bytes(msg.data))

            self.saved += 1
            self.get_logger().info(f"saved {self.saved}/{self.max_saves}: {os.path.basename(fname)}")
        except Exception as e:
            self.get_logger().error(f"save failed: {e}")
            return

        if self.saved >= self.max_saves:
            self.get_logger().info("Reached max_saves, shutting down.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()