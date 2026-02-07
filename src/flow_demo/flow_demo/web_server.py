#!/usr/bin/env python3
import threading
import time

import numpy as np
import cv2
from flask import Flask, Response, request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


def rosimg_to_bgr(msg: Image) -> np.ndarray:
    """Minimal conversion without cv_bridge. Supports rgb8/bgr8/mono8."""
    h, w = msg.height, msg.width
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    if msg.encoding == "mono8":
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported encoding: {msg.encoding}")


class MJPEGServer(Node):
    def __init__(self):
        super().__init__("web_server")
        self.declare_parameter("topic", "/flow/vis")
        self.declare_parameter("port", 8765)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("max_fps", 30.0)

        self.topic = self.get_parameter("topic").value
        self.port = int(self.get_parameter("port").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.max_fps = float(self.get_parameter("max_fps").value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(Image, self.topic, self.cb, qos)

        self._lock = threading.Lock()
        self._latest_bgr = None
        self._latest_stamp = time.time()

        self.get_logger().info(f"Subscribing to {self.topic}")
        self.get_logger().info(f"Serving MJPEG on 0.0.0.0:{self.port}")

        self.app = Flask(__name__)

        @self.app.get("/")
        def index():
            return f"""
            <html><body style="font-family: sans-serif">
              <h3>ROS2 MJPEG Viewer</h3>
              <p>Topic: <b>{self.topic}</b></p>
              <img src="/stream.mjpg" />
            </body></html>
            """

        @self.app.get("/stream.mjpg")
        def stream():
            return Response(self._gen(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        # run flask in a thread so rclpy can spin
        self._flask_thread = threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=self.port, threaded=True, debug=False),
            daemon=True,
        )
        self._flask_thread.start()

    def cb(self, msg: Image):
        try:
            bgr = rosimg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")
            return
        with self._lock:
            self._latest_bgr = bgr
            self._latest_stamp = time.time()

    def _gen(self):
        last_sent = 0.0
        while rclpy.ok():
            with self._lock:
                frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                stamp = self._latest_stamp

            if frame is None:
                time.sleep(0.05)
                continue

            # cap fps
            now = time.time()
            if self.max_fps > 0:
                min_dt = 1.0 / self.max_fps
                if now - last_sent < min_dt:
                    time.sleep(min_dt - (now - last_sent))
            last_sent = time.time()

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"X-Timestamp: " + str(stamp).encode() + b"\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")

def main():
    rclpy.init()
    node = MJPEGServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()