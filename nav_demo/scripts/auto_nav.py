#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time

import actionlib
import rospy
import tf
import tf.transformations as tft
import yaml
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String, Int32
from std_srvs.srv import SetBool, Trigger


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def normalize_angle(a):
    """角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quat(yaw):
    """绕 Z 轴旋转 yaw 的四元数 (w, x, y, z)。"""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def is_near(px, py, x, y, tol=0.05):
    return abs(px - x) <= tol and abs(py - y) <= tol


# ---------------------------------------------------------------------------
# TF 位姿读取：全局唯一 listener，预热后查询快且准
# ---------------------------------------------------------------------------
class PoseTracker:
    def __init__(self, map_frame="map", base_frame="base_link"):
        self.map_frame = map_frame
        self.base_frame = base_frame
        self._listener = tf.TransformListener()   # 整个节点只建一个 listener

    def warmup(self, timeout=5.0):
        try:
            return self._listener.waitForTransform(
                self.map_frame, self.base_frame,
                rospy.Time(0), rospy.Duration(timeout))
        except Exception as e:
            rospy.logwarn("TF 预热失败: %s", e)
            return False

    def get_pose(self):
        """返回 (x, y, yaw)；拿不到时返回 None。"""
        try:
            trans, rot = self._listener.lookupTransform(
                self.map_frame, self.base_frame, rospy.Time(0))
            _, _, yaw = tft.euler_from_quaternion(rot)
            return trans[0], trans[1], yaw
        except tf.Exception:
            return None


# ---------------------------------------------------------------------------
# 任务路点表：坐标 + 任务类型(没有 task 字段的 waypoint 会自动到这里查)
#   红绿灯 -> 等绿灯才通行; count -> 人群/立牌计数; plate -> 车牌识别
# ---------------------------------------------------------------------------
TASK_YAW_TOLERANCE = 0.20
TASK_POINTS = [
    {"x": 0.8,   "y": 1.7,   "yaw": 3.1416,  "task": "traffic_light", "name": "红绿灯1"},
    {"x": -1.1, "y": 1.7,  "yaw": 4.7124,  "task": "count",         "name": "识别1"},
    {"x": -1.75,  "y": 1.07,  "yaw": 0.0,     "task": "count",         "name": "识别2"},
    {"x": -0.8,  "y": 0.45,  "yaw": 1.5708,  "task": "count",         "name": "识别3"},
    {"x": -0.8,  "y": 0.45,  "yaw": -1.5708, "task": "count",         "name": "识别4"},
    {"x": -0.06, "y": -0.2, "yaw": 3.1416,  "task": "count",         "name": "识别5"},
    {"x": -0.06, "y": -0.03, "yaw": 4.7124,  "task": "traffic_light", "name": "红绿灯2"},
    {"x": 1.1,   "y": -1.75, "yaw": 0.0,     "task": "plate",         "name": "车牌1"},
    {"x": 1.1,   "y": -0.68, "yaw": 0.0,     "task": "plate",         "name": "车牌2"},
]


# ---------------------------------------------------------------------------
# 导航 + 视觉主类
# ---------------------------------------------------------------------------
class AutoNavigator:
    def __init__(self):
        # ---------- 参数 ----------
        self.waypoints_file = rospy.get_param(
            "~waypoints_file",
            rospy.get_param("waypoints_file", os.path.expanduser("~/waypoints.yaml")))
        self.auto_start = rospy.get_param("~auto_start", False)

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.vision_start_srv = rospy.get_param("~vision_start_service", "/vision_start")
        self.vision_stop_srv = rospy.get_param("~vision_stop_service", "/vision_stop")
        self.vision_green_srv = rospy.get_param("~vision_wait_green_service", "/vision_wait_green")

        # 旋转控制参数(转向要灵敏: 比例增益拉高, 上限对齐局部规划器的 max_vel_theta=1.4)
        self.rot_kp = rospy.get_param("~rot_kp", 3.0)
        self.rot_max = rospy.get_param("~rot_max_speed", 1.4)
        self.rot_min = rospy.get_param("~rot_min_speed", 0.25)
        self.rot_deadband = rospy.get_param("~rot_deadband", 0.02)
        self.rot_timeout = rospy.get_param("~rot_timeout", 15.0)
        self.move_timeout = rospy.get_param("~move_timeout", 120.0)
        self.settle_time = rospy.get_param("~settle_time", 0.5)
        # 离目标近于此距离时跳过"转向行进方向"(方位角已无意义)。
        # 默认对齐 move_base 的 xy_goal_tolerance, 到了这个距离本来就算到位了。
        self.path_align_min_dist = rospy.get_param("~path_align_min_dist", 0.15)

        # 视觉停留时间(waypoints.yaml 的 stay_time_at_bc 会覆盖)
        self.stay_time = rospy.get_param("~stay_time_at_bc", 3.0)

        # 分街区累计: 每个识别点记 (总人数, 社区人员数, 非社区人员数)
        self.street_tally = [
            {"name": "1街区", "points": ["识别1", "识别2", "识别3"], "data": {}},
            {"name": "2街区", "points": ["识别4", "识别5"], "data": {}},
        ]

        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.pose = PoseTracker(self.map_frame, self.base_frame)

        # ---------- 视觉服务(耐心等待, 最多 vision_wait_timeout 秒) ----------
        self.vision_ok = False
        wait_timeout = rospy.get_param("~vision_wait_timeout", 60.0)
        rospy.loginfo("等待视觉服务, 最多 %.0f 秒 ...", wait_timeout)
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < wait_timeout:
            try:
                rospy.wait_for_service(self.vision_start_srv, timeout=0.5)
                rospy.wait_for_service(self.vision_stop_srv, timeout=0.5)
                rospy.wait_for_service(self.vision_green_srv, timeout=0.5)
                self.start_vision = rospy.ServiceProxy(self.vision_start_srv, SetBool)
                self.stop_vision = rospy.ServiceProxy(self.vision_stop_srv, Trigger)
                self.wait_green = rospy.ServiceProxy(self.vision_green_srv, Trigger)
                self.vision_ok = True
                rospy.loginfo("视觉服务连接成功")
                break
            except rospy.ROSException:
                rospy.loginfo_throttle(5.0, "视觉服务还没就绪, 继续等待 ...")
                rospy.sleep(1.0)
        if not self.vision_ok:
            rospy.logwarn("等待视觉服务超时, 本脚本只导航、不做识别任务")

        # ---------- 订阅视觉结果(用于打印) ----------
        self.last_result = None
        self.last_count = None
        self.last_plate = None
        self.last_member = None      # 社区人员数
        self.last_stranger = None    # 非社区人员数
        self.last_names = None       # 识别到的社区人员姓名
        rospy.Subscriber("/vision/result", String, self.on_result)
        rospy.Subscriber("/vision/crowd_count", Int32, self.on_count)
        rospy.Subscriber("/vision/plate_text", String, self.on_plate)
        rospy.Subscriber("/vision/member_count", Int32, self.on_member)
        rospy.Subscriber("/vision/stranger_count", Int32, self.on_stranger)
        rospy.Subscriber("/vision/member_names", String, self.on_names)

    # ---------------- 视觉结果回调 ----------------
    def on_result(self, msg):
        self.last_result = msg.data

    def on_count(self, msg):
        self.last_count = msg.data

    def on_plate(self, msg):
        self.last_plate = msg.data

    def on_member(self, msg):
        self.last_member = msg.data

    def on_stranger(self, msg):
        self.last_stranger = msg.data

    def on_names(self, msg):
        self.last_names = msg.data

    # ---------------- 基础动作 ----------------
    def stop_robot(self):
        twist = Twist()
        for _ in range(3):
            self.vel_pub.publish(twist)
            rospy.sleep(0.05)

    def cancel_navigation(self, timeout=1.0):
        """取消 move_base 当前目标, 等它真的松手后再清零 cmd_vel。

        """
        try:
            self.client.cancel_all_goals()
        except Exception:
            pass
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < timeout:
            # PENDING/ACTIVE = move_base 还在控车, 变 PREEMPTED 才算放手
            if self.client.get_state() not in (GoalStatus.PENDING,
                                               GoalStatus.ACTIVE):
                break
            rospy.sleep(0.02)
        self.stop_robot()

    def rotate_to_yaw(self, target_yaw, timeout=None, max_speed=None):
        """闭环原地旋转到 target_yaw：最短弧 + 比例控制 + 平滑减速 + 双向纠偏。"""
        self.cancel_navigation()
        timeout = self.rot_timeout if timeout is None else timeout
        max_speed = self.rot_max if max_speed is None else max_speed
        target_yaw = normalize_angle(target_yaw)

        twist = Twist()
        rate = rospy.Rate(30)
        start = time.time()

        while not rospy.is_shutdown():
            if time.time() - start > timeout:
                rospy.logwarn("旋转超时(%.1fs)，强制停止", timeout)
                break
            pose = self.pose.get_pose()
            if pose is None:
                rate.sleep()
                continue
            current_yaw = pose[2]
            err = normalize_angle(target_yaw - current_yaw)
            if abs(err) <= self.rot_deadband:
                rospy.loginfo("旋转到位: 当前 yaw=%.4f, 目标 %.4f, 误差 %.4f",
                              current_yaw, target_yaw, abs(err))
                break
            speed = max(self.rot_min, min(max_speed, self.rot_kp * abs(err)))
            twist.angular.z = math.copysign(speed, err)
            self.vel_pub.publish(twist)
            rate.sleep()

        self.stop_robot()
        rospy.sleep(0.1)

    def align_to_path(self, x, y):
        """出发前先原地转到朝向下一点的方位, 让车一启动就是前向行驶。

        目标已经很近时(例如只换朝向、位置不变的两个识别点)方位角没有意义, 直接跳过,
        后面的 rotate_to_yaw(yaw) 会负责把朝向摆正。
        """
        pose = self.pose.get_pose()
        if pose is None:
            rospy.logwarn("拿不到位姿, 跳过转向行进方向")
            return
        dx = x - pose[0]
        dy = y - pose[1]
        dist = math.hypot(dx, dy)
        if dist <= self.path_align_min_dist:
            return
        bearing = math.atan2(dy, dx)
        err = abs(normalize_angle(bearing - pose[2]))
        rospy.loginfo("先转向行进方向 %.1f°(当前偏差 %.1f°, 距离 %.2f m)",
                      math.degrees(bearing), math.degrees(err), dist)
        self.rotate_to_yaw(bearing)

    def move_to(self, x, y):
        """只把机器人送到 (x, y)：goal 朝向=当前朝向, 不让 move_base 凑 yaw。"""
        pose_now = self.pose.get_pose()
        if pose_now is None:
            rospy.logerr("无法获取当前位姿，不下发导航目标")
            return False
        heading = pose_now[2]

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        qw, qx, qy, qz = yaw_to_quat(heading)
        goal.target_pose.pose.orientation.w = qw
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz

        rospy.loginfo("移动至 (%.2f, %.2f) ...", x, y)
        self.client.send_goal(goal)
        finished = self.client.wait_for_result(rospy.Duration(self.move_timeout))
        if not finished:
            rospy.logwarn("移动超时(%.0fs)，取消本段目标", self.move_timeout)
            self.client.cancel_goal()
            self.stop_robot()
            return False
        if self.client.get_state() != GoalStatus.SUCCEEDED:
            rospy.logerr("move_base 结果异常, state=%s", self.client.get_state())
            self.stop_robot()
            return False
        return True

    def navigate_to(self, x, y, yaw):
        """1) 转到行进方向  2) 送到位置  3) 对齐该点要求的 yaw。"""
        self.align_to_path(x, y)
        if not self.move_to(x, y):
            return False
        rospy.sleep(self.settle_time)
        self.rotate_to_yaw(yaw)
        return True

    # ---------------- 视觉任务 ----------------
    def current_task_point(self, wp):
        """返回匹配的任务点; YAML 有 task 时优先使用。"""
        if wp.get("task"):
            return {"task": wp["task"], "name": wp.get("name", "")}
        wp_yaw = wp.get("yaw")
        for t in TASK_POINTS:
            if not is_near(wp["x"], wp["y"], t["x"], t["y"]):
                continue
            if (wp_yaw is None or
                    abs(normalize_angle(float(wp_yaw) - t["yaw"])) <= TASK_YAW_TOLERANCE):
                return t
        return None

    def current_task(self, wp):
        point = self.current_task_point(wp)
        return point["task"] if point else None

    def report_count(self, wp):
        """打印识别点结果, 并累计到所在街区。"""
        total = int(self.last_count) if self.last_count is not None else 0
        member = int(self.last_member) if self.last_member is not None else 0
        stranger = int(self.last_stranger) if self.last_stranger is not None else 0
        names = (self.last_names or "").strip()

        task_point = self.current_task_point(wp) or {}
        name = task_point.get("name", "") or "识别"

        if member + stranger > 0:
            rospy.loginfo(">>> %s: 识别到社区 %d 人，非社区 %d 人（共 %d 人）",
                          name, member, stranger, total)
            if names:
                rospy.loginfo("    社区人员: %s", names)
        else:
            # 没启用社区人员识别(vision_node 里 ENABLE_REID=False 或图库缺失)
            rospy.loginfo(">>> %s人数 = %d（未启用社区人员识别, result: %s）",
                          name, total, self.last_result)
            names = ""

        for street in self.street_tally:
            if name not in street["points"]:
                continue
            street["data"][name] = (total, member, stranger)
            if name != street["points"][-1]:
                return
            # 该街区最后一个识别点跑完 -> 汇总(只报人数, 不报人员姓名)
            t = sum(v[0] for v in street["data"].values())
            m = sum(v[1] for v in street["data"].values())
            s = sum(v[2] for v in street["data"].values())
            detail = " + ".join(
                "%s %d人" % (p, street["data"][p][0])
                for p in street["points"] if p in street["data"])
            rospy.loginfo("=" * 56)
            rospy.loginfo("在%s共识别到 %d 人（%s）", street["name"], t, detail)
            if m + s > 0:
                rospy.loginfo("在%s：识别到社区 %d 人，非社区 %d 人", street["name"], m, s)
            rospy.loginfo("=" * 56)
            return

    def do_vision_task(self, wp, task):
        """识别/车牌路点: 开启识别 -> 停留 -> 关闭 -> 打印结果。"""
        self.last_result = self.last_count = self.last_plate = None
        self.last_member = self.last_stranger = self.last_names = None
        if self.vision_ok:
            try:
                self.start_vision(True)
            except rospy.ServiceException as e:
                rospy.logwarn("调用 %s 失败: %s", self.vision_start_srv, e)
        rospy.loginfo("在任务点停留 %.1f 秒做识别 ...", self.stay_time)
        rospy.sleep(self.stay_time)
        if self.vision_ok:
            try:
                self.stop_vision()
            except rospy.ServiceException:
                pass

        if task == "count":
            self.report_count(wp)
        else:  # plate
            rospy.loginfo(">>> 车牌 = %s (result: %s)", self.last_plate, self.last_result)

    def do_wait_green(self, wp):
        """红绿灯路点: 停车等绿灯, 绿灯才继续(绿灯才通行)。"""
        if not self.vision_ok:
            rospy.logerr("视觉服务不可用, 无法判断红绿灯; 停留 %.1fs 后继续", self.stay_time)
            rospy.sleep(self.stay_time)
            return
        rospy.loginfo("红绿灯: 停车等待绿灯 ...")
        while not rospy.is_shutdown():
            try:
                res = self.wait_green()
            except rospy.ServiceException as e:
                rospy.logwarn("调用 %s 失败: %s", self.vision_green_srv, e)
                rospy.sleep(1.0)
                continue
            if res.success:
                rospy.loginfo("绿灯确认, 出发!")
                return
            rospy.logwarn("本次等待超时, 继续等绿灯 ...")

    # ---------------- 初始位姿 / 定位 ----------------
    def set_initial_pose(self, x, y, yaw):
        pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped,
                              queue_size=1, latch=True)
        rospy.sleep(0.5)
        for _ in range(3):                   # 多发几次, 防止 amcl 没收到
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = self.map_frame
            msg.header.stamp = rospy.Time.now()
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = 0.0
            qw, qx, qy, qz = yaw_to_quat(yaw)
            msg.pose.pose.orientation.w = qw
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            pub.publish(msg)
            rospy.sleep(0.3)
        rospy.loginfo("已发布初始位姿 (%.2f, %.2f, yaw=%.3f)", x, y, yaw)

    def wait_localized(self, x0, y0, timeout=10.0):
        rospy.loginfo("等待定位/位姿可用 ...")
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < timeout:
            p = self.pose.get_pose()
            if p is not None:
                rospy.loginfo("当前位姿 x=%.3f y=%.3f yaw=%.3f", p[0], p[1], p[2])
                if math.hypot(p[0] - x0, p[1] - y0) > 1.0:
                    rospy.logwarn("当前位姿与起点偏差较大, 请确认定位/初始位姿正确!")
                return True
            rospy.sleep(0.2)
        rospy.logwarn("等待位姿超时, 继续执行(若导航失败请检查 map->base_link TF)")
        return False

    # ---------------- 主流程 ----------------
    def run(self):
        if not os.path.exists(self.waypoints_file):
            rospy.logerr("找不到 waypoints 文件: %s", self.waypoints_file)
            return

        with open(self.waypoints_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        waypoints = data["waypoints"]
        self.stay_time = data.get("stay_time_at_bc", self.stay_time)

        init = data.get("initial_pose", {})
        ix = float(init.get("x", 1.7))
        iy = float(init.get("y", 1.7))
        iyaw = float(init.get("yaw", math.pi))

        rospy.loginfo("共 %d 个导航点 | stay_time=%.1fs", len(waypoints), self.stay_time)

        # 连接 move_base
        rospy.loginfo("等待 move_base ...")
        if not self.client.wait_for_server(rospy.Duration(30.0)):
            rospy.logfatal("连不上 move_base! 请先启动导航(amcl + move_base)")
            return
        rospy.loginfo("move_base 已连接")

        # 初始位姿(起点) + 定位等待
        self.set_initial_pose(ix, iy, iyaw)
        self.pose.warmup(5.0)
        self.wait_localized(ix, iy, timeout=10.0)

        if not self.auto_start:
            try:
                input(">>> 按回车键开始自动导航 ...")
            except EOFError:
                rospy.logwarn("由 roslaunch 启动, 无交互终端, 直接开始")

        for idx, wp in enumerate(waypoints):
            x = float(wp["x"])
            y = float(wp["y"])
            yaw = float(wp["yaw"])
            task = self.current_task(wp)
            rospy.loginfo("\n===== 第 %d/%d 点: (%.2f, %.2f) yaw=%.4f | 任务: %s =====",
                          idx + 1, len(waypoints), x, y, yaw, task)

            ok = self.navigate_to(x, y, yaw)
            if not ok:
                rospy.logerr("第 %d 个点导航失败, 中止任务", idx + 1)
                break
            rospy.loginfo("已到达第 %d 个点", idx + 1)

            if task == "traffic_light":
                self.do_wait_green(wp)
            elif task in ("count", "plate"):
                self.do_vision_task(wp, task)

            rospy.sleep(0.3)

        rospy.loginfo("\n全部导航任务完成!")
        self.cancel_navigation()
        self.stop_robot()


def main():
    rospy.init_node("auto_nav_vision", anonymous=True)
    AutoNavigator().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
