#!/usr/bin/env python3
# 功能：数画面里 person 的数量（零训练，用官方预训练模型）
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

CLASS_NAME = "person"   # 数哪个类就写哪个
CONF = 0.5              # 低于此置信度不数（不准就调低到0.3）

class CountNode:
    def __init__(self):
        rospy.init_node("count_node")
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")   # 官方预训练，首次运行自动下载
        self.pub_img = rospy.Publisher("~annotated", Image, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.cb,
                         queue_size=1, buff_size=2**24)

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False)[0]
        n = sum(1 for b in results.boxes
                if results.names[int(b.cls[0])] == CLASS_NAME
                and float(b.conf[0]) >= CONF)
        rospy.loginfo("数量: %d", n)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(results.plot(), "bgr8"))

if __name__ == "__main__":
    CountNode()
    rospy.spin()