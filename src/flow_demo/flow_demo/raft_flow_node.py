#!/usr/bin/env python3
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


# -----------------------------
# Utilities: ROS Image <-> numpy
# -----------------------------
def rosimg_to_rgb8(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image with encoding rgb8 into (H,W,3) uint8."""
    if msg.encoding.lower() != "rgb8":
        raise ValueError(f"Expected rgb8, got {msg.encoding}")

    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    expected = h * w * 3
    if arr.size != expected:
        raise ValueError(f"rgb8 size mismatch: got {arr.size}, expected {expected}")
    return arr.reshape((h, w, 3))


def rgb8_to_rosimg(rgb: np.ndarray, stamp_msg, frame_id: str, topic_encoding="rgb8") -> Image:
    """Encode (H,W,3) uint8 to sensor_msgs/Image."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    h, w, _ = rgb.shape
    msg = Image()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = topic_encoding
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = rgb.tobytes()
    return msg


def flow32_to_rosimg(flow: np.ndarray, stamp_msg, frame_id: str) -> Image:
    """
    Encode flow (H,W,2) float32 into sensor_msgs/Image with encoding 32FC2.
    This is convenient for downstream nodes that want raw flow.
    """
    assert flow.dtype == np.float32 and flow.ndim == 3 and flow.shape[2] == 2
    h, w, _ = flow.shape
    msg = Image()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "32FC2"
    msg.is_bigendian = 0
    msg.step = w * 2 * 4  # 2 channels * float32
    msg.data = flow.tobytes()
    return msg


# -----------------------------
# Flow visualization (simple HSV)
# -----------------------------
def flow_to_vis_rgb(flow: np.ndarray, clip: float = 30.0) -> np.ndarray:
    """
    Convert flow (H,W,2) to RGB visualization (H,W,3) uint8.
    - Hue = direction
    - Value = magnitude (clipped)
    """
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)  # [-pi, pi]
    hue = (ang + np.pi) / (2 * np.pi)  # [0,1]

    mag = np.clip(mag / max(clip, 1e-6), 0.0, 1.0)
    sat = np.ones_like(mag, dtype=np.float32)
    val = mag.astype(np.float32)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)  # (H,W,3)

    # HSV -> RGB
    # Minimal implementation to avoid extra deps:
    import colorsys
    h, w, _ = hsv.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = colorsys.hsv_to_rgb(float(hsv[y, x, 0]), float(hsv[y, x, 1]), float(hsv[y, x, 2]))
            rgb[y, x, 0] = int(r * 255)
            rgb[y, x, 1] = int(g * 255)
            rgb[y, x, 2] = int(b * 255)
    return rgb


# -----------------------------
# RAFT loader wrapper
# -----------------------------
@dataclass
class RaftConfig:
    raft_repo: str
    weights_path: str
    device: str = "cpu"
    iters: int = 12



def make_raft_args(cfg):
    # fill the fields RAFT expects
    return argparse.Namespace(
        small=getattr(cfg, "small", False),
        mixed_precision=getattr(cfg, "mixed_precision", False),
        dropout=getattr(cfg, "dropout", 0.0),
        alternate_corr=getattr(cfg, "alternate_corr", False),
    )

class RaftWrapper:
    def __init__(self, cfg: RaftConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Make RAFT importable
        sys.path.insert(0, cfg.raft_repo)

        # RAFT expects: from core.raft import RAFT, plus config args
        RAFT_ROOT = "/workspaces/flow-ros2/third_party/RAFT"
        RAFT_CORE = os.path.join(RAFT_ROOT, "core")

# make RAFT importable (core.raft) and also allow "import update" inside core/raft.py
        if RAFT_ROOT not in sys.path:
            sys.path.insert(0, RAFT_ROOT)
        if RAFT_CORE not in sys.path:
            sys.path.insert(0, RAFT_CORE)

        from core.raft import RAFT
        from core.utils.utils import InputPadder

        self.InputPadder = InputPadder

      # in raft_flow_node.py (RaftWrapper / config creation)


        self.args = make_raft_args(cfg)
        self.model = RAFT(self.args)

        ckpt = torch.load(cfg.weights_path, map_location="cpu")
        # many RAFT checkpoints store as {'state_dict': ...}
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # strip "module." if present
        state = {k.replace("module.", ""): v for k, v in state.items()}

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray, iters: Optional[int] = None) -> np.ndarray:
        """
        Inputs: RGB uint8 images (H,W,3)
        Output: flow float32 (H,W,2) in pixel units at the input resolution
        """
        iters = iters or self.cfg.iters

        # RAFT expects torch float in [0,255] shaped (B,3,H,W)
        t1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).float()[None].to(self.device)
        t2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).float()[None].to(self.device)

        padder = self.InputPadder(t1.shape)
        t1, t2 = padder.pad(t1, t2)

        flow_low, flow_up = self.model(t1, t2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_up)[0]  # (2,H,W)
        flow = flow.permute(1, 2, 0).contiguous().float().cpu().numpy()
        return flow


# -----------------------------
# ROS2 Node
# -----------------------------
class RaftFlowNode(Node):
    def __init__(self):
        super().__init__("raft_flow_node")

        # Parameters
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("vis_topic", "/flow/vis")
        self.declare_parameter("raw_topic", "/flow/raw")
        self.declare_parameter("publish_raw", True)

        self.declare_parameter("raft_repo", "/workspaces/flow-ros2/third_party/RAFT")
        self.declare_parameter("weights_path", "/workspaces/flow-ros2/weights/raft-sintel.pth")
        self.declare_parameter("device", "cpu")  # later: cuda
        self.declare_parameter("iters", 12)
        self.declare_parameter("vis_clip", 30.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.vis_topic = self.get_parameter("vis_topic").value
        self.raw_topic = self.get_parameter("raw_topic").value
        self.publish_raw = bool(self.get_parameter("publish_raw").value)

        raft_repo = self.get_parameter("raft_repo").value
        weights_path = self.get_parameter("weights_path").value
        device = self.get_parameter("device").value
        iters = int(self.get_parameter("iters").value)

        self.vis_clip = float(self.get_parameter("vis_clip").value)

        if not os.path.exists(weights_path):
            self.get_logger().error(f"RAFT weights not found: {weights_path}")
        if not os.path.isdir(raft_repo):
            self.get_logger().error(f"RAFT repo not found: {raft_repo}")

        self.raft = RaftWrapper(RaftConfig(
            raft_repo=raft_repo,
            weights_path=weights_path,
            device=device,
            iters=iters,
        ))

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.pub_vis = self.create_publisher(Image, self.vis_topic, qos)
        self.pub_raw = self.create_publisher(Image, self.raw_topic, qos)

        self.prev_rgb: Optional[np.ndarray] = None
        self.prev_stamp = None

        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Publishing vis: {self.vis_topic}")
        if self.publish_raw:
            self.get_logger().info(f"Publishing raw: {self.raw_topic} (32FC2)")

    def on_image(self, msg: Image):
        try:
            rgb = rosimg_to_rgb8(msg)
        except Exception as e:
            self.get_logger().error(f"Decode failed: {e}")
            return

        if self.prev_rgb is None:
            self.prev_rgb = rgb
            self.prev_stamp = msg.header.stamp
            return

        # Run flow on prev->current
        flow = self.raft.infer(self.prev_rgb, rgb)

        # Publish vis
        vis = flow_to_vis_rgb(flow, clip=self.vis_clip)
        vis_msg = rgb8_to_rosimg(vis, msg.header.stamp, msg.header.frame_id or "camera", topic_encoding="rgb8")
        self.pub_vis.publish(vis_msg)

        # Publish raw flow
        if self.publish_raw:
            raw_msg = flow32_to_rosimg(flow.astype(np.float32), msg.header.stamp, msg.header.frame_id or "camera")
            self.pub_raw.publish(raw_msg)

        # Update prev
        self.prev_rgb = rgb
        self.prev_stamp = msg.header.stamp


def main():
    rclpy.init()
    node = RaftFlowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()