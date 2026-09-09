#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import rospy
import cv2
import numpy as np
import hyperlpr3 as lpr3
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO

# ========================== 可调参数 ==========================
IMAGE_TOPIC   = "/camera/image_raw"      # 相机图像话题(仿真里通常是这个)
MODEL_PATH    = "yolov8n.pt"             # 官方预训练模型, 首次运行自动下载
PERSON_CLASS  = "person"                 # 计数类别(卡通立牌可改成你的类名)
CONF_PERSON   = 0.5                      # 计数置信度
CONF_PLATE    = 0.6                      # 车牌置信度
CONF_LIGHT    = 0.4                      # 红绿灯检测置信度
NEED_GREEN    = 5                        # 连续几帧绿灯才确认"可以通行"
GREEN_TIMEOUT = 30.0                     # 等绿灯最多多少秒
MAX_FRAMES    = 20                       # 一次识别最多处理帧数(防漏停死循环)
STABLE_FRAMES = 3                        # 连续几帧结果一致就自动结束
TASK_RADIUS   = 0.25                     # 距路点多近算"到达"

# 任务路点表: 按你给的坐标, 只有带识别任务的点才列出
TASK_WAYPOINTS = [
    {"name": "traffic_light_1", "x": 2.0,  "y": 2.5,   "task": "traffic_light"},
    {"name": "count_1",         "x": -1.5, "y": 0.65,   "task": "count"},   # 识别1
    {"name": "count_2",         "x": -1.5, "y": 0.65,   "task": "count"},   # 识别2(同点朝反方向, 任务相同)
    {"name": "traffic_light_2", "x": -0.15, "y": 0.65,   "task": "traffic_light"},
    {"name": "plate_1",         "x": 1.7,  "y": -2.3,  "task": "plate"},   # 车牌1
    {"name": "plate_2",         "x": 1.4,  "y": -1.15, "task": "plate"},   # 车牌2
]
# ========================== 可调参数 ==========================


_FONT_CACHE = {}
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _load_cn_font(size):
    """找一个支持中文的字体(带缓存)"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                continue
    return None


def draw_cn_text(img, text, pos, size=22, color=(255, 0, 0)):
    """在 BGR 图上画支持中文的文字, 返回新图。找不到中文字体时退回 cv2(中文会显示?)"""
    if not text:
        return img
    font = _load_cn_font(size)
    if font is None or not _PIL_OK:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text((int(pos[0]), int(pos[1])), text, font=font,
              fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class VisionRecognition:
    def __init__(self):
        rospy.init_node("vision_recognition")
        self.bridge = CvBridge()

        rospy.loginfo("正在加载模型(首次运行会自动下载, 请耐心等待)...")
        self.model = YOLO(MODEL_PATH)                 # 人群计数 + 红绿灯检测
        self.catcher = lpr3.LicensePlateCatcher()     # 车牌识别, 自带检测模型
        rospy.loginfo("模型加载完成")

        # 运行状态
        self.enabled = False       # 是否处于识别中
        self.force_task = rospy.get_param("~force_task", "")  # 调试: count/plate/traffic_light
        self.continuous_display = rospy.get_param("~continuous_display", True)  # /vision/annotated 是否一直有画面
        self.frames = 0            # 本次已处理帧数
        self.stable = 0            # 结果连续相同的帧数
        self.last_key = None       # 上一个结果 key

        # 位姿(判断当前在哪个路点)
        self.pose = None
        self.amcl_ok = False        # 是否收到过 /amcl_pose
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.on_pose)  # move_base+amcl
        rospy.Subscriber("/odom", Odometry, self.on_odom)          # 纯odom导航(仅无amcl时兜底)

        # 发布器
        self.pub_result = rospy.Publisher("/vision/result", String, queue_size=1)
        self.pub_light  = rospy.Publisher("/vision/traffic_light_state", String, queue_size=1)
        self.pub_count  = rospy.Publisher("/vision/crowd_count", Int32, queue_size=1)
        self.pub_plate  = rospy.Publisher("/vision/plate_text", String, queue_size=1)
        self.pub_img    = rospy.Publisher("/vision/annotated", Image, queue_size=1)

        # 服务
        rospy.Service("/vision_start", SetBool, self.handle_start)
        rospy.Service("/vision_stop", Trigger, self.handle_stop)
        rospy.Service("/vision_wait_green", Trigger, self.handle_wait_green)

        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback,
                         queue_size=1, buff_size=2**24)
        rospy.loginfo("视觉识别节点已启动, 等待指令...")

    # ---------------- 位姿 ----------------
    def on_pose(self, msg):
        # amcl 给的是地图坐标, 与 TASK_WAYPOINTS 一致, 优先使用
        self.amcl_ok = True
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def on_odom(self, msg):
        # 只有从来没收到过 amcl 位姿时才用 odom 兜底(odom 是里程计坐标, 通常不等于地图坐标)
        if not self.amcl_ok:
            self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def current_task(self):
        """按机器人坐标判断当前路点任务: count / plate / traffic_light / none"""
        if self.force_task:
            return self.force_task
        if self.pose is None:
            rospy.logwarn_throttle(3.0, "还没收到 /amcl_pose 或 /odom, 无法判断路点")
            return "none"
        best = None
        for wp in TASK_WAYPOINTS:
            d = ((wp["x"] - self.pose[0]) ** 2 + (wp["y"] - self.pose[1]) ** 2) ** 0.5
            if d <= TASK_RADIUS and (best is None or d < best[0]):
                best = (d, wp["task"])
        return best[1] if best else "none"

    # ---------------- 服务处理 ----------------
    def handle_start(self, req):
        self.enabled = bool(req.data)
        if self.enabled:
            self.frames = 0
            self.stable = 0
            self.last_key = None
            rospy.loginfo("开启识别, 当前任务: %s", self.current_task())
        else:
            rospy.loginfo("关闭识别")
        return SetBoolResponse(success=True, message="OK")

    def handle_stop(self, req):
        self.enabled = False
        rospy.loginfo("识别已停止")
        return TriggerResponse(success=True, message="stopped")

    def handle_wait_green(self, req):
        """阻塞直到连续 NEED_GREEN 帧绿灯(绿灯才通行)。控制器在红绿灯路点调用它。"""
        rospy.loginfo("红绿灯路点: 开始等待绿灯...")
        green_n = 0
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < GREEN_TIMEOUT:
            try:
                img_msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=1.0)
            except rospy.ROSException:
                continue
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            color = self.detect_traffic_light(frame, publish=True)   # 等灯时也发画框画面到 /vision/annotated
            self.pub_light.publish(String(color))
            green_n = green_n + 1 if color == "green" else 0
            if green_n >= NEED_GREEN:
                rospy.loginfo("已确认绿灯, 可以通行")
                return TriggerResponse(success=True, message="green, go")
        rospy.logwarn("等待绿灯超时(%s秒)", GREEN_TIMEOUT)
        return TriggerResponse(success=False, message="timeout")

    # ---------------- 三个识别功能 ----------------
    def count_people(self, frame, publish=True):
        """识别1/识别2: 数画面里的人/立牌"""
        results = self.model(frame, verbose=False)[0]
        n = 0
        for b in results.boxes:
            if results.names[int(b.cls[0])] == PERSON_CLASS and float(b.conf[0]) >= CONF_PERSON:
                n += 1
        if publish:
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(results.plot(), "bgr8"))
        return n

    def read_plate(self, frame):
        """车牌识别: 返回画面里所有置信度达标的车牌, 每项 (code, conf, box=[x1,y1,x2,y2])"""
        plates = []
        for plate in self.catcher(frame):
            code, conf = str(plate[0]), float(plate[1])
            if conf >= CONF_PLATE:
                box = plate[3] if len(plate) > 3 else None
                plates.append((code, conf, box))
        plates.sort(key=lambda p: p[1], reverse=True)   # 按置信度从高到低, 输出顺序稳定
        return plates

    def draw_plate_boxes(self, frame, plates):
        """在画面里给每块车牌画蓝框+车牌号, 返回标注后的图"""
        ann = frame.copy()
        y0 = 30
        for code, conf, box in plates:
            try:
                x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
                cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 0, 0), 2)
                ann = draw_cn_text(ann, code, (x1, max(20, y1 - 8)), 22, (255, 0, 0))
            except Exception:
                pass
            ann = draw_cn_text(ann, "plate: " + code, (10, y0), 22, (255, 0, 0))
            y0 += 30
        return ann

    def draw_color_boxes(self, frame, color):
        """把红/黄/绿亮斑用对应颜色框出来(仿真红绿灯的灯就是一团亮斑)"""
        if color == "none" or frame is None:
            return frame
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        if color == "red":
            mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 80, 100), (10, 255, 255)),
                                  cv2.inRange(hsv, (156, 80, 100), (180, 255, 255)))
        elif color == "yellow":
            mask = cv2.inRange(hsv, (11, 80, 100), (34, 255, 255))
        elif color == "green":
            mask = cv2.inRange(hsv, (35, 80, 100), (85, 255, 255))
        else:
            return frame
        bgr = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}[color]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 25:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, color, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
        return frame

    def judge_light_color(self, crop):
        """在红绿灯框内判断亮灯颜色"""
        if crop is None or crop.size == 0:
            return "none"
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masks = {
            "red": cv2.bitwise_or(cv2.inRange(hsv, (0, 80, 120), (10, 255, 255)),
                                  cv2.inRange(hsv, (156, 80, 120), (180, 255, 255))),
            "yellow": cv2.inRange(hsv, (11, 80, 120), (34, 255, 255)),
            "green": cv2.inRange(hsv, (35, 80, 120), (85, 255, 255)),
        }
        counts = {c: cv2.countNonZero(m) for c, m in masks.items()}
        color = max(counts, key=counts.get)
        return color if counts[color] >= 15 else "none"

    def detect_color_full_frame(self, frame):
        """兜底: 不依赖YOLO, 在全图找最明显的红/黄/绿亮点(仿真红绿灯通常很显眼)"""
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        masks = {
            "red": cv2.bitwise_or(cv2.inRange(hsv, (0, 80, 100), (10, 255, 255)),
                                  cv2.inRange(hsv, (156, 80, 100), (180, 255, 255))),
            "yellow": cv2.inRange(hsv, (11, 80, 100), (34, 255, 255)),
            "green": cv2.inRange(hsv, (35, 80, 100), (85, 255, 255)),
        }
        counts = {c: cv2.countNonZero(m) for c, m in masks.items()}
        color = max(counts, key=counts.get)
        return color if counts[color] >= 40 else "none"

    def detect_traffic_light(self, frame, publish=True):
        """红绿灯1/红绿灯2: 返回 red/yellow/green/none"""
        results = self.model(frame, verbose=False)[0]
        color = "none"
        best = None
        for b in results.boxes:
            if results.names[int(b.cls[0])] != "traffic light":
                continue
            conf = float(b.conf[0])
            if best is None or conf > best[1]:
                best = (b, conf)
        if best is not None and best[1] >= CONF_LIGHT:
            x1, y1, x2, y2 = [int(v) for v in best[0].xyxy[0]]
            color = self.judge_light_color(frame[max(0, y1):y2, max(0, x1):x2])
        if color == "none":
            color = self.detect_color_full_frame(frame)   # YOLO没认出时, 全图找彩色亮点
        if publish:
            ann = results.plot()
            ann = self.draw_color_boxes(ann, color)   # 给亮斑画框
            cv2.putText(ann, "light: " + color, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if color == "green" else (0, 0, 255), 3)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        return color

    # ---------------- 主回调: 按路点任务识别 ----------------
    def image_callback(self, msg):
        if not self.enabled:
            # 省CPU持续显示: 空闲时只转发原始画面(不跑检测), 检测只在任务进行时跑
            if self.continuous_display:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            return
        if self.frames >= MAX_FRAMES:
            rospy.logwarn("本次识别达到最大帧数仍无稳定结果, 自动结束(请检查是否对准目标)")
            self.enabled = False
            self.frames = 0
            return
        self.frames += 1
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        task = self.current_task()

        if task == "count":
            n = self.count_people(frame)
            self.pub_count.publish(Int32(n))
            if n > 0:
                self.update_stable("count", str(n))
            else:
                self.reset_stable("count")

        elif task == "plate":
            plates = self.read_plate(frame)
            if plates:
                joined = " | ".join(p[0] for p in plates)   # 多块车牌用 | 分隔
                ann = self.draw_plate_boxes(frame, plates)
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
                self.pub_plate.publish(String(joined))
                self.update_stable("plate", joined)
            else:
                self.reset_stable("plate")

        elif task == "traffic_light":
            color = self.detect_traffic_light(frame)
            self.pub_light.publish(String(color))
            if color == "green":
                self.update_stable("traffic_light", "green")
            else:
                self.reset_stable("traffic_light")

        else:
            # 位姿匹配不到路点时, 兜底识别(计数+车牌)。auto_nav 本就在任务点才调用 /vision_start。
            rospy.logwarn_throttle(3.0, "位姿未匹配到任务点, 使用兜底识别(count+plate)")
            n = self.count_people(frame)
            self.pub_count.publish(Int32(n))
            if n > 0:
                self.update_stable("count", str(n))
            else:
                self.reset_stable("count")
            plates = self.read_plate(frame)
            if plates:
                joined = " | ".join(p[0] for p in plates)
                ann = self.draw_plate_boxes(frame, plates)
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
                self.pub_plate.publish(String(joined))
                self.update_stable("plate", joined)
            else:
                self.reset_stable("plate")

    def update_stable(self, task, value):
        key = "%s=%s" % (task, value)
        if key == self.last_key:
            self.stable += 1
        else:
            self.last_key = key
            self.stable = 1
        if self.stable >= STABLE_FRAMES:
            self.finish(task, value)

    def reset_stable(self, task):
        if self.last_key is not None and self.last_key.startswith(task + "="):
            self.last_key = None
            self.stable = 0

    def finish(self, task, value):
        self.enabled = False
        self.frames = 0
        self.stable = 0
        self.last_key = None
        self.pub_result.publish(String("task=%s result=%s" % (task, value)))
        rospy.loginfo("识别完成 -> task=%s result=%s (已自动关闭, 等下一次 /vision_start)", task, value)


if __name__ == "__main__":
    VisionRecognition()
    rospy.spin()
