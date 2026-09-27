#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人员重识别 ROS 节点 —— 判断画面里的人是不是社区人员。

接口设计对齐 nav_demo/scripts/vision_node.py 的既有约定, 方便和 auto_nav 对接。

订阅:
    /camera/image_raw           (sensor_msgs/Image)  相机画面

发布:
    /reid/annotated             (sensor_msgs/Image)  画了框和姓名的画面
    /reid/detections            (std_msgs/String)    JSON, 每个人的识别结果
    /reid/member_present        (std_msgs/Bool)      画面里是否有社区人员

服务:
    /reid_start      (std_srvs/SetBool)  开关"连续识别"模式
    /reid_identify   (std_srvs/Trigger)  单次识别: 抓当前帧跑一遍并返回结果
    /reid_reload     (std_srvs/Trigger)  重新加载图库(不用重启节点)

参数(都能在 launch 里覆盖):
    ~image_topic      默认 /camera/image_raw
    ~gallery          图库 .npz 路径(空则用 ~/.ros/reid_gallery.npz)
    ~weights          OSNet 权重路径
    ~detector_weights YOLOv8 权重
    ~threshold        判定阈值, 见 calibrate_threshold.py
    ~conf_det         人框置信度
    ~min_person_h     人框最小高度(像素)
    ~tta_flip         是否翻转增强(默认 true)
    ~threads          torch CPU 线程数(2核机器建议 2)
    ~continuous       启动后是否直接进入连续识别
"""

import json
import os
import sys
import threading
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_engine import (ReIDEngine, DEFAULT_WEIGHTS, DEFAULT_DETECTOR,  # noqa: E402
                         draw_detections)

# 连续模式下最多多久处理一帧(秒)。CPU 上跑不动相机帧率, 主动限流避免积压。
MIN_INTERVAL = 0.2


class PersonReIDNode(object):
    def __init__(self):
        rospy.init_node("person_reid")

        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        gallery = rospy.get_param("~gallery", "")
        if not gallery:
            gallery = os.path.join(os.path.expanduser("~"), ".ros", "reid_gallery.npz")
        self.gallery_path = gallery
        self.threshold = float(rospy.get_param("~threshold", 0.6))
        self.continuous = bool(rospy.get_param("~continuous", False))

        rospy.loginfo("正在加载 ReID 模型(首次运行较慢)...")
        self.engine = ReIDEngine(
            weights=rospy.get_param("~weights", DEFAULT_WEIGHTS),
            detector_weights=rospy.get_param("~detector_weights", DEFAULT_DETECTOR),
            conf_det=float(rospy.get_param("~conf_det", 0.5)),
            min_person_h=int(rospy.get_param("~min_person_h", 40)),
            tta_flip=bool(rospy.get_param("~tta_flip", True)),
            threads=int(rospy.get_param("~threads", 0)) or None,
        )
        rospy.loginfo("模型加载完成")

        self.bridge = CvBridge()
        self.lock = threading.Lock()     # 模型不是线程安全的, 推理必须串行
        self.busy = False
        self.enabled = self.continuous
        self.last_run = 0.0
        self.latest_msg = None

        self.load_gallery(self.gallery_path, quiet=True)

        self.pub_img = rospy.Publisher("/reid/annotated", Image, queue_size=1)
        self.pub_det = rospy.Publisher("/reid/detections", String, queue_size=1)
        self.pub_member = rospy.Publisher("/reid/member_present", Bool, queue_size=1)

        rospy.Service("/reid_start", SetBool, self.handle_start)
        rospy.Service("/reid_identify", Trigger, self.handle_identify)
        rospy.Service("/reid_reload", Trigger, self.handle_reload)

        rospy.Subscriber(self.image_topic, Image, self.image_callback,
                         queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("人员重识别节点已启动, 阈值=%.2f, 监听 %s"
                      % (self.threshold, self.image_topic))

    # ---------------- 图库 ----------------
    def load_gallery(self, path, quiet=False):
        if not path or not os.path.exists(path):
            if not quiet:
                rospy.logwarn("图库不存在: %s (所有人都会被判为陌生人)" % path)
            return 0
        try:
            n = self.engine.load_gallery(path)
            names = sorted(set(self.engine.gallery_names))
            rospy.loginfo("图库已加载: %d 条特征, %d 个人 %s"
                          % (n, len(names), names))
            return n
        except Exception as e:
            rospy.logerr("图库加载失败: %s" % e)
            return 0

    # ---------------- 服务 ----------------
    def handle_start(self, req):
        self.enabled = bool(req.data)
        rospy.loginfo("连续识别: %s" % ("开启" if self.enabled else "关闭"))
        return SetBoolResponse(success=True, message="OK")

    def handle_identify(self, req):
        """单次识别: 抓最近一帧跑一遍, 结果放在 message 里返回。

        message 格式:  "members=张三,李四; strangers=2; total=4"
        控制器可以直接解析, 或者去订阅 /reid/detections 拿完整 JSON。
        """
        msg = self.latest_msg
        if msg is None:
            return TriggerResponse(success=False, message="还没收到图像")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return TriggerResponse(success=False, message="图像转换失败: %s" % e)
        results = self.process(frame, publish=True)
        members = [r["name"] for r in results if r["is_member"]]
        strangers = sum(1 for r in results if not r["is_member"])
        summary = "members=%s; strangers=%d; total=%d" % (
            ",".join(sorted(set(members))) if members else "", strangers, len(results))
        return TriggerResponse(success=True, message=summary)

    def handle_reload(self, req):
        n = self.load_gallery(self.gallery_path)
        if n == 0:
            return TriggerResponse(success=False, message="图库为空或加载失败")
        return TriggerResponse(success=True, message="reloaded %d features" % n)

    # ---------------- 主流程 ----------------
    def image_callback(self, msg):
        self.latest_msg = msg
        if not self.enabled:
            return
        # 限流: CPU 上跑不到相机帧率, 保证同一时刻只有一次推理在跑
        now = time.time()
        if self.busy or (now - self.last_run) < MIN_INTERVAL:
            return
        self.busy = True
        self.last_run = now
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process(frame, publish=True)
        except Exception as e:
            rospy.logerr_throttle(5.0, "识别出错: %s" % e)
        finally:
            self.busy = False

    def process(self, frame, publish=True):
        """跑一次完整识别, 返回 results(每人的框 / 姓名 / 相似度)"""
        with self.lock:
            t0 = time.time()
            results = self.engine.identify(frame, threshold=self.threshold)
            dt = time.time() - t0

        if not publish:
            return results

        # 汇总画面里出现的社区人员姓名
        members = sorted(set(r["name"] for r in results if r["is_member"]))
        payload = {
            "stamp": time.time(),
            "cost_ms": round(dt * 1000, 1),
            "threshold": self.threshold,
            "members": members,
            "strangers": sum(1 for r in results if not r["is_member"]),
            "detections": [
                {"box": [round(v, 1) for v in r["box"]],
                 "det_conf": round(r["det_conf"], 3),
                 "is_member": r["is_member"],
                 "name": r["name"],
                 "sim": None if r["sim"] is None else round(r["sim"], 4),
                 "topk": [[n, round(s, 4)] for n, s in r["topk"]]}
                for r in results],
        }
        self.pub_det.publish(String(json.dumps(payload, ensure_ascii=False)))
        self.pub_member.publish(Bool(bool(members)))
        ann = draw_detections(frame, results)   # 画一张 720p 要几十毫秒, 所以放在发布分支里
        try:
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5.0, "发布标注图失败: %s" % e)
        return results


if __name__ == "__main__":
    try:
        PersonReIDNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
