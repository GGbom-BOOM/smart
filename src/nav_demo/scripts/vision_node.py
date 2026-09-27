#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
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
# 权重实际放在哪: 按顺序找, 找到就用现成的 —— 否则 ultralytics 只会看当前
# 工作目录, 换个目录启动就会重新下一遍(而且 GitHub 在这边时通时不通)。
WEIGHT_DIRS   = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "person_reid", "models"),
                 os.path.expanduser("~/.ros"),
                 os.path.expanduser("~/models")]
PERSON_CLASS  = "person"                 # 计数类别(卡通立牌可改成你的类名)
CONF_PERSON   = 0.5                      # 计数置信度
CONF_PLATE    = 0.6                      # 车牌置信度
CONF_LIGHT    = 0.4                      # 红绿灯检测置信度
NEED_GREEN    = 5                        # 连续几帧绿灯才确认"可以通行"
GREEN_TIMEOUT = 30.0                     # 等绿灯最多多少秒
MAX_FRAMES    = 20                       # 一次识别最多处理帧数(防漏停死循环)
STABLE_FRAMES = 3                        # 连续几帧结果一致就自动结束
TASK_RADIUS     = 0.25                   # 距路点多近算"到达"
TASK_YAW_RADIUS = 0.35                   # 朝向匹配容差(rad)
PROCESS_HZ      = 4.0                    # 识别时每秒最多处理几帧(0=不限速)
GREEN_HZ        = 10.0                   # 等绿灯时每秒最多判几帧(0=不限速)
VIEW_HZ         = 8.0                    # 空闲时 /vision/annotated 每秒最多转发几帧(0=不限速)

# ========================== 社区人员识别 ==========================
# 识别点上除了数人, 还判断每个人是不是本社区登记人员(见 person_reid 包)。
ENABLE_REID     = True     # 关掉就退回"只数人"的老行为
REID_THRESHOLD  = 0.6      # 相似度阈值。务必先用 person_reid/scripts/calibrate_threshold.py 标定!
REID_MIN_H      = 40       # 人框最小高度(像素), 比这小的框看不清外观, 不计入
REID_TTA_FLIP   = True     # 翻转增强(更准, 耗时约一倍)
REID_THREADS    = 1        # torch CPU 线程数: 本机只有 2 核, 留一个核给导航/Gazebo。
                           # 实测 1 线程不但不慢(少了线程调度开销), 导航也不卡了
# ========================== 社区人员识别 ==========================

# 任务路点表: 只有需要识别/等待绿灯的点才列出
# 同坐标不同朝向的点必须用 yaw 区分, 例如识别3/4和识别5
TASK_WAYPOINTS = [
    {"name": "traffic_light_1", "x": 0.8,   "y": 1.7,   "yaw": 3.1416,  "task": "traffic_light"},
    {"name": "count_1",         "x": -1.1, "y": 1.7,  "yaw": 4.7124,  "task": "count"},   # 识别1
    {"name": "count_2",         "x": -1.75,  "y": 1.07,  "yaw": 0.0,     "task": "count"},   # 识别2
    {"name": "count_3",         "x": -1.0,  "y": 0.45,  "yaw": 1.5708,  "task": "count"},   # 识别3
    {"name": "count_4",         "x": -1.0,  "y": 0.45,  "yaw": -1.5708, "task": "count"},   # 识别4
    {"name": "count_5",         "x": -0.06, "y": -0.1, "yaw": 3.1416,  "task": "count"},   # 识别5
    {"name": "traffic_light_2", "x": -0.06, "y": -0.03, "yaw": 4.7124,  "task": "traffic_light"},
    {"name": "plate_1",         "x": 1.1,   "y": -1.75, "yaw": 0.0,     "task": "plate"},   # 车牌1
    {"name": "plate_2",         "x": 1.1,   "y": -0.68, "yaw": 0.0,     "task": "plate"},   # 车牌2
]
# ========================== 可调参数 ==========================


def _resolve_weight(name):
    """把权重文件名换成实际路径。

    先看当前目录, 再依次翻 WEIGHT_DIRS —— 都找不到就把原名还回去, 让
    ultralytics 走它自己的下载流程(和以前行为一致)。
    """
    if os.path.isabs(name) or os.sep in name:
        return name
    if os.path.exists(name):
        return name
    for d in WEIGHT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            rospy.loginfo("复用已有权重: %s" % os.path.normpath(p))
            return os.path.normpath(p)
    return name


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


# ========================== 红绿灯取色 ==========================
# 红/黄/绿的 HSV 阈值。红色跨色环两端, 所以要两段拼起来。
# S 下限固定 80; V 下限按场景给: 整图找亮斑用 100, 已经框住的灯用 120。
_HSV_S_MIN = 80
_HSV_SPANS = {
    "red":    (((0, _HSV_S_MIN, 0), (10, 255, 255)),
               ((156, _HSV_S_MIN, 0), (180, 255, 255))),
    "yellow": (((11, _HSV_S_MIN, 0), (34, 255, 255)),),
    "green":  (((35, _HSV_S_MIN, 0), (85, 255, 255)),),
}
_BGR_OF = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}


def light_masks(bgr, v_min):
    """把 BGR 图变成 {颜色: 二值掩码} —— 模糊和 HSV 转换只做一次, 三色共用。

    以前全图兜底、框内判色、画亮斑框各写了一遍这套阈值, 改一处容易漏另外两处。
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    masks = {}
    for name, spans in _HSV_SPANS.items():
        for lo, hi in spans:
            m = cv2.inRange(hsv, (lo[0], lo[1], v_min), hi)
            masks[name] = m if name not in masks else cv2.bitwise_or(masks[name], m)
    return masks


def dominant_light(masks, min_px):
    """像素最多的颜色; 不足 min_px 个像素就当没看到灯"""
    counts = {c: cv2.countNonZero(m) for c, m in masks.items()}
    color = max(counts, key=counts.get)
    return color if counts[color] >= min_px else "none"


def draw_blobs(bgr, color, masks, min_area=25):
    """把指定颜色的连通块框出来(仿真红绿灯的灯就是一团亮斑)。原地改并返回"""
    mask = masks.get(color) if (masks and color != "none") else None
    if mask is None or color not in _BGR_OF:
        return bgr
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), _BGR_OF[color], 2)
        cv2.putText(bgr, color, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BGR_OF[color], 2)
    return bgr


class VisionRecognition:
    def __init__(self):
        rospy.init_node("vision_recognition")
        self.bridge = CvBridge()

        rospy.loginfo("正在加载模型(首次运行会自动下载, 请耐心等待)...")
        self.model = YOLO(_resolve_weight(MODEL_PATH))  # 人群计数 + 红绿灯检测
        self.catcher = lpr3.LicensePlateCatcher()     # 车牌识别, 自带检测模型
        self._draw_detections = None
        self.reid = self.load_reid()                  # 社区人员识别(加载失败则为 None)
        rospy.loginfo("模型加载完成")

        # 运行状态
        self.enabled = False       # 是否处于识别中
        self.frames = 0            # 本次已处理帧数
        self.stable = 0            # 结果连续相同的帧数
        self.last_key = None       # 上一个结果 key
        self._img_sub = None       # 相机订阅句柄(按需创建/释放)
        self.latest_frame = None   # 最近一帧(cv_bridge 给的是视图, 不额外拷贝)
        self.latest_stamp = None   # 最近一帧的时间戳
        self._last_proc = 0.0      # 上一次识别处理的时刻(限速用)
        self._last_view = 0.0      # 上一次转发标注画面的时刻(限速用)
        self._waiting_green = 0    # >0 表示有线程在等绿灯(此时必须保持画面)
        self._green_stamp = None   # 等绿灯时用来避免把同一帧重复数成好几帧

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
        # 社区人员识别结果
        self.pub_member   = rospy.Publisher("/vision/member_count", Int32, queue_size=1)
        self.pub_stranger = rospy.Publisher("/vision/stranger_count", Int32, queue_size=1)
        self.pub_names    = rospy.Publisher("/vision/member_names", String, queue_size=1)

        # 服务
        rospy.Service("/vision_start", SetBool, self.handle_start)
        rospy.Service("/vision_stop", Trigger, self.handle_stop)
        rospy.Service("/vision_wait_green", Trigger, self.handle_wait_green)

        self._update_subscription()                     # 初始为空闲 -> 不订阅
        # 每秒看一次要不要订阅(比如中途有人打开 image_view 看画面)
        self._watchdog_timer = rospy.Timer(rospy.Duration(1.0), self.on_watchdog)
        rospy.loginfo("视觉识别节点已启动, 等待指令...")

    # ---------------- 相机订阅(按需) ----------------
    def _has_viewer(self):
        """有没有人在看 /vision/annotated。没人看就不用生成、发布标注画面。"""
        try:
            return self.pub_img.get_num_connections() > 0
        except Exception:
            return False

    def _update_subscription(self):
        """按需订阅/退订相机。

        空闲时(没在识别、没等绿灯、没人看画面)退订, 这一路 1280x720@30 的
        图像就不进 Python 了 —— 省下来的 CPU 给 move_base 和 Gazebo, 导航
        会明显顺一些。需要画面时随时重订, 一两帧之内就能拿到图。
        """
        need = bool(self.enabled or self._waiting_green > 0 or self._has_viewer())
        if need and self._img_sub is None:
            self._img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback,
                                             queue_size=1, buff_size=2**24)
            rospy.loginfo("订阅相机 %s", IMAGE_TOPIC)
        elif not need and self._img_sub is not None:
            self._img_sub.unregister()
            self._img_sub = None
            self.latest_frame = None
            self.latest_stamp = None
            rospy.loginfo("空闲: 已退订相机(不占 CPU), 需要时自动重订")

    def on_watchdog(self, _event=None):
        self._update_subscription()

    # ---------------- 社区人员识别 ----------------
    def load_reid(self):
        """加载社区人员识别引擎(person_reid 包)。

        任何一步失败都返回 None, 让节点退回"只数人"的老行为 —— 不能因为
        图库或权重缺一个, 就把整个视觉节点(红绿灯/车牌)搞挂。
        """
        if not ENABLE_REID:
            rospy.loginfo("社区人员识别已关闭(ENABLE_REID=False), 只数人数")
            return None

        # 定位 person_reid 包的 scripts 目录
        scripts_dir = ""
        try:
            import rospkg
            scripts_dir = os.path.join(rospkg.RosPack().get_path("person_reid"), "scripts")
        except Exception:
            pass
        if not scripts_dir or not os.path.isdir(scripts_dir):
            scripts_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "person_reid", "scripts"))
        if not os.path.isdir(scripts_dir):
            rospy.logwarn("找不到 person_reid 包(试过 %s), 本次只数人数", scripts_dir)
            return None
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        try:
            from reid_engine import ReIDEngine, draw_detections
            self._draw_detections = draw_detections   # 画中文姓名标签用
        except ImportError as e:
            rospy.logwarn("导入 reid_engine 失败(%s), 本次只数人数", e)
            return None

        gallery = os.path.join(os.path.dirname(scripts_dir), "gallery", "gallery.npz")
        if not os.path.exists(gallery):
            rospy.logwarn("找不到图库 %s, 本次只数人数(所有人都会显示为「陌生人」)", gallery)
            rospy.logwarn("建库: python3 %s/build_gallery.py --src <参考图目录> --out %s",
                          scripts_dir, gallery)
            return None

        try:
            engine = ReIDEngine(tta_flip=REID_TTA_FLIP, min_person_h=REID_MIN_H,
                                threads=REID_THREADS or None)
            n = engine.load_gallery(gallery)
            people = sorted(set(engine.gallery_names))
            rospy.loginfo("社区人员识别已启用: %d 条特征 / %d 人(%s), 阈值 %.2f",
                          n, len(people), "、".join(people), REID_THRESHOLD)
            if REID_THRESHOLD <= 0.6:
                rospy.logwarn("当前阈值 %.2f 偏低, 容易把陌生人误判为社区人员。"
                              "建议跑 calibrate_threshold.py 标定", REID_THRESHOLD)
            return engine
        except Exception as e:
            rospy.logwarn("社区人员识别初始化失败(%s), 本次只数人数", e)
            return None

    # ---------------- 位姿 ----------------
    @staticmethod
    def _yaw_from_quaternion(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def on_pose(self, msg):
        # amcl 给的是地图坐标, 与 TASK_WAYPOINTS 一致, 优先使用
        self.amcl_ok = True
        pose = msg.pose.pose
        self.pose = (pose.position.x, pose.position.y,
                     self._yaw_from_quaternion(pose.orientation))

    def on_odom(self, msg):
        # 只有从来没收到过 amcl 位姿时才用 odom 兜底(odom 是里程计坐标, 通常不等于地图坐标)
        if not self.amcl_ok:
            pose = msg.pose.pose
            self.pose = (pose.position.x, pose.position.y,
                         self._yaw_from_quaternion(pose.orientation))

    def current_task(self):
        """按机器人坐标和朝向判断当前路点任务: count / plate / traffic_light / none"""
        if self.pose is None:
            rospy.logwarn_throttle(3.0, "还没收到 /amcl_pose 或 /odom, 无法判断路点")
            return "none"
        best = None
        for wp in TASK_WAYPOINTS:
            d = ((wp["x"] - self.pose[0]) ** 2 + (wp["y"] - self.pose[1]) ** 2) ** 0.5
            dyaw = abs(math.atan2(
                math.sin(self.pose[2] - wp["yaw"]),
                math.cos(self.pose[2] - wp["yaw"])))
            if (d <= TASK_RADIUS and dyaw <= TASK_YAW_RADIUS
                    and (best is None or d < best[0])):
                best = (d, wp["task"])
        return best[1] if best else "none"

    # ---------------- 服务处理 ----------------
    def handle_start(self, req):
        self.enabled = bool(req.data)
        if self.enabled:
            self.frames = 0
            self.stable = 0
            self.last_key = None
            self._last_proc = 0.0            # 立刻处理第一帧, 不等限速窗口
            self._update_subscription()
            rospy.loginfo("开启识别, 当前任务: %s", self.current_task())
        else:
            rospy.loginfo("关闭识别")
            self._update_subscription()      # 停了就退订, 把 CPU 让给导航
        return SetBoolResponse(success=True, message="OK")

    def handle_stop(self, req):
        self.enabled = False
        self._update_subscription()          # 导航马上就要走, 立刻退订
        rospy.loginfo("识别已停止")
        return TriggerResponse(success=True, message="stopped")

    def handle_wait_green(self, req):
        """阻塞直到连续 NEED_GREEN 帧绿灯(绿灯才通行)。控制器在红绿灯路点调用它。

        画面来自常驻回调留下的最新帧 —— 老写法在循环里反复 wait_for_message,
        每一轮都要重新和 Gazebo 的相机建一次连接再收一张 720p 大图, 又慢又
        费 CPU, 现在已经不需要了。
        """
        rospy.loginfo("红绿灯路点: 开始等待绿灯...")
        green_n = 0
        t0 = time.time()
        self._waiting_green += 1
        self._green_stamp = None
        self._update_subscription()          # 保证有画面
        try:
            while not rospy.is_shutdown() and time.time() - t0 < GREEN_TIMEOUT:
                frame, stamp = self.latest_frame, self.latest_stamp
                if frame is None or stamp == self._green_stamp:
                    rospy.sleep(0.05)        # 还没等到新帧: 别把同一帧重复数
                    continue
                self._green_stamp = stamp
                color = self.detect_traffic_light(frame, publish=self._has_viewer())
                self.pub_light.publish(String(color))
                green_n = green_n + 1 if color == "green" else 0
                if green_n >= NEED_GREEN:
                    rospy.loginfo("已确认绿灯, 可以通行")
                    return TriggerResponse(success=True, message="green, go")
                if GREEN_HZ > 0:             # 等灯时限速, 别为了等灯把 CPU 占满
                    rospy.sleep(1.0 / GREEN_HZ)
        finally:
            self._waiting_green -= 1
            self._update_subscription()
        rospy.logwarn("等待绿灯超时(%s秒)", GREEN_TIMEOUT)
        return TriggerResponse(success=False, message="timeout")

    # ---------------- 三个识别功能 ----------------
    def draw_summary(self, img, info):
        """在画面左上角写一行汇总: 社区x人 / 非社区y人"""
        if info["reid"]:
            txt = "社区 %d 人 / 非社区 %d 人 (共 %d 人)" % (
                info["member"], info["stranger"], info["total"])
            if info["names"]:
                txt += "  " + "、".join(info["names"])
        else:
            txt = "共 %d 人 (未启用社区人员识别)" % info["total"]
        return draw_cn_text(img, txt, (10, 34), 26, (0, 255, 255))

    def publish_member_info(self, info):
        """把社区人员统计发到话题上, 供 auto_nav 打印"""
        self.pub_member.publish(Int32(info["member"]))
        self.pub_stranger.publish(Int32(info["stranger"]))
        self.pub_names.publish(String("、".join(info["names"])))

    def count_people(self, frame, publish=True):
        """识别点: 数画面里的人/立牌, 并区分社区人员与非社区人员。

        返回 {"total": 总人数, "member": 社区人员数, "stranger": 非社区人员数,
              "names": [识别到的社区人员姓名], "reid": 是否启用了社区人员识别}
        启用 ReID 时, 检测+提特征+比对在 ReIDEngine 里一次做完, 不会多跑一遍 YOLO。
        """
        if self.reid is None:
            # 没启用或加载失败: 退回纯计数, 全部当"未知"处理
            res = self.model(frame, verbose=False)[0]
            n = sum(1 for b in res.boxes
                    if res.names[int(b.cls[0])] == PERSON_CLASS
                    and float(b.conf[0]) >= CONF_PERSON)
            if publish and self._has_viewer():
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(res.plot(), "bgr8"))
            return {"total": n, "member": 0, "stranger": 0,
                    "names": [], "reid": False}

        results = self.reid.identify(frame, threshold=REID_THRESHOLD)
        members = [r for r in results if r["is_member"]]
        info = {"total": len(results), "member": len(members),
                "stranger": len(results) - len(members),
                "names": sorted(set(r["name"] for r in members)), "reid": True}
        if publish:
            self.publish_member_info(info)          # 数值结果必须发, 导航要用
            if self._has_viewer():                  # 没人看就不画, 画一张 720p 要几十毫秒
                ann = self._draw_detections(frame, results)   # 每人一个中文姓名标签
                ann = self.draw_summary(ann, info)            # 左上角汇总
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        return info

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

    def judge_light_color(self, crop):
        """在红绿灯框内判断亮灯颜色。框内的灯是小块亮斑, 所以门槛比整图高一点"""
        if crop is None or crop.size == 0:
            return "none"
        return dominant_light(light_masks(crop, 120), 15)

    def detect_traffic_light(self, frame, publish=True):
        """红绿灯1/红绿灯2: 返回 red/yellow/green/none"""
        results = self.model(frame, verbose=False)[0]
        best = None
        for b in results.boxes:
            if results.names[int(b.cls[0])] != "traffic light":
                continue
            conf = float(b.conf[0])
            if best is None or conf > best[1]:
                best = (b, conf)

        color = "none"
        if best is not None and best[1] >= CONF_LIGHT:
            x1, y1, x2, y2 = [int(v) for v in best[0].xyxy[0]]
            color = self.judge_light_color(frame[max(0, y1):y2, max(0, x1):x2])

        # 整图亮斑只在"要兜底判色"或"要画图"时才算: 720p 的 blur+HSV 不便宜,
        # 而算一次就能同时给出兜底颜色和要画的框(以前是两个函数各算一遍)。
        masks = light_masks(frame, 100) if (color == "none" or publish) else None
        if color == "none":
            color = dominant_light(masks, 40)   # YOLO没认出时, 全图找彩色亮点
        if publish:
            ann = draw_blobs(results.plot(), color, masks)
            cv2.putText(ann, "light: " + color, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if color == "green" else (0, 0, 255), 3)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        return color

    # ---------------- 主回调: 按路点任务识别 ----------------
    def _view_due(self):
        """空闲转发画面也限速(看一眼不需要 30fps, 省下的都是导航的)"""
        if VIEW_HZ <= 0:
            return True
        now = time.time()
        if now - self._last_view < 1.0 / VIEW_HZ:
            return False
        self._last_view = now
        return True

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # 存下来给等绿灯用(cv_bridge 返回的是视图, 这行不拷贝图像)
        self.latest_frame = frame
        self.latest_stamp = msg.header.stamp

        if not self.enabled:
            # 省CPU: 空闲不跑检测; 有人看 /vision/annotated 才低频转发原始画面
            if self._has_viewer() and self._view_due():
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            return
        if self.frames >= MAX_FRAMES:
            rospy.logwarn("本次识别达到最大帧数仍无稳定结果, 自动结束(请检查是否对准目标)")
            self.enabled = False
            self.frames = 0
            self._update_subscription()
            return
        if PROCESS_HZ > 0 and time.time() - self._last_proc < 1.0 / PROCESS_HZ:
            return          # 限速窗口内: 丢掉这帧, 让出 CPU 给 move_base/Gazebo
        self._last_proc = time.time()
        self.frames += 1
        task = self.current_task()
        if task == "none":
            # 位姿匹配不到路点时兜底: 计数和车牌都做。auto_nav 本就在任务点才调 /vision_start。
            rospy.logwarn_throttle(3.0, "位姿未匹配到任务点, 使用兜底识别(count+plate)")
            tasks = ("count", "plate")
        else:
            tasks = (task,)
        for t in tasks:
            self._run_task(t, frame)

    def _run_task(self, task, frame):
        """跑一个任务的一帧: 结果既发话题, 也拿去判定"连续几帧一致"。"""
        if task == "count":
            info = self.count_people(frame)
            self.pub_count.publish(Int32(info["total"]))
            # 用"总数/社区/非社区"一起做稳定判据: 三个数都稳定了才算定稿
            value = None
            if info["total"] > 0:
                value = "%d:%d:%d" % (info["total"], info["member"], info["stranger"])
            self._mark_stable("count", value)

        elif task == "plate":
            plates = self.read_plate(frame)
            joined = " | ".join(p[0] for p in plates) if plates else None  # 多块车牌用 | 分隔
            if joined:
                if self._has_viewer():
                    ann = self.draw_plate_boxes(frame, plates)
                    self.pub_img.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
                self.pub_plate.publish(String(joined))
            self._mark_stable("plate", joined)

        elif task == "traffic_light":
            color = self.detect_traffic_light(frame, publish=self._has_viewer())
            self.pub_light.publish(String(color))
            self._mark_stable("traffic_light", "green" if color == "green" else None)

    def _mark_stable(self, task, value):
        """value 为 None 表示这一帧没识别出东西 -> 清掉稳定计数重新攒"""
        if value is None:
            self.reset_stable(task)
        else:
            self.update_stable(task, value)

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
        self._update_subscription()      # 识别结束: 退订相机, 把 CPU 还给导航
        # count 的 value 内部格式是 "总数:社区:非社区", 对外换成好读的形式
        text = value
        if task == "count" and value.count(":") == 2:
            total, member, stranger = value.split(":")
            text = "总%s人(社区%s/非社区%s)" % (total, member, stranger)
        self.pub_result.publish(String("task=%s result=%s" % (task, text)))
        rospy.loginfo("识别完成 -> task=%s result=%s (已自动关闭, 等下一次 /vision_start)",
                      task, text)


if __name__ == "__main__":
    VisionRecognition()
    rospy.spin()
