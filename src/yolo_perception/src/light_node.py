#!/usr/bin/env python3
# 功能：红绿灯识别 + 绿灯才通行（零训练）
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO

CONF = 0.4       # 检测灯的置信度阈值
NEED = 5         # 连续几帧同色才切换（防抖）
SPEED = 0.3      # 前进速度 m/s，按你的车调
MIN_PIX = 15     # 灯区域最少要有多少个“亮色像素”才算数

class LightNode:
    def __init__(self):
        rospy.init_node("light_node")
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")   # 官方预训练，含 traffic light 类
        self.pub_img = rospy.Publisher("~annotated", Image, queue_size=1)
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.green_n = 0
        self.other_n = 0
        self.go = False
        rospy.Subscriber("/usb_cam/image_raw", Image, self.cb,
                         queue_size=1, buff_size=2**24)

    def judge_color(self, crop):
        """在红绿灯框内判断亮灯颜色：返回 'red'/'yellow'/'green'/'none'"""
        if crop is None or crop.size == 0:
            return "none"
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # HSV范围：红(两段) 黄 绿；V>120只保留“亮着的灯”，排除暗灯壳
        masks = {
            "red":    cv2.bitwise_or(cv2.inRange(hsv, (0, 80, 120), (10, 255, 255)),
                                     cv2.inRange(hsv, (156, 80, 120), (180, 255, 255))),
            "yellow": cv2.inRange(hsv, (11, 80, 120), (34, 255, 255)),
            "green":  cv2.inRange(hsv, (35, 80, 120), (85, 255, 255)),
        }
        counts = {c: cv2.countNonZero(m) for c, m in masks.items()}
        color = max(counts, key=counts.get)
        return color if counts[color] >= MIN_PIX else "none"

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False)[0]

        color = "none"
        for b in results.boxes:
            if results.names[int(b.cls[0])] != "traffic light":
                continue
            if float(b.conf[0]) < CONF:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            color = self.judge_color(frame[y1:y2, x1:x2])
            break   # 只判断最可信的那一个灯

        # 决策：连续 NEED 帧绿灯 -> 走；红/黄/没看到灯 -> 停
        if color == "green":
            self.green_n += 1; self.other_n = 0
        else:
            self.other_n += 1; self.green_n = 0
        if self.green_n >= NEED:
            self.go = True
        elif self.other_n >= NEED:
            self.go = False

        tw = Twist()
        tw.linear.x = SPEED if self.go else 0.0
        self.pub_cmd.publish(tw)

        # 在图上标颜色，方便看效果
        annotated = results.plot()
        cv2.putText(annotated, color, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0) if self.go else (0, 0, 255), 3)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))
        rospy.loginfo("灯=%s 状态=%s", color, "GO" if self.go else "STOP")

if __name__ == "__main__":
    LightNode()
    rospy.spin()