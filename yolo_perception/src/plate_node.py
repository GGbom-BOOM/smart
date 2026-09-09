#!/usr/bin/env python3
# 功能：中文车牌识别（零训练，HyperLPR3 开源模型）
import rospy
import cv2
import hyperlpr3 as lpr3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CONF = 0.6   # 置信度低于此值不打印（太乱就调高）

class PlateNode:
    def __init__(self):
        rospy.init_node("plate_node")
        self.bridge = CvBridge()
        self.catcher = lpr3.LicensePlateCatcher()  # 首次运行自动下载模型
        rospy.Subscriber("/usb_cam/image_raw", Image, self.cb,
                         queue_size=1, buff_size=2**24)

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        for plate in self.catcher(frame):
            # plate = [车牌号, 置信度, 类型, 框坐标]
            code, conf = plate[0], plate[1]
            if conf >= CONF:
                rospy.loginfo("车牌: %s (置信度 %.2f)", code, conf)

if __name__ == "__main__":
    PlateNode()
    rospy.spin()