import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class DummyImagePub(Node):
    def __init__(self):
        super().__init__('dummy_image_pub')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.5, self.tick)  # 2 Hz
        self.i = 0

    def tick(self):
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (self.i % 255)  # simple changing pattern

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img.tobytes()

        self.pub.publish(msg)
        self.get_logger().info(f"Published frame {self.i} -> /camera/image_raw")
        self.i += 1

def main():
    rclpy.init()
    node = DummyImagePub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()