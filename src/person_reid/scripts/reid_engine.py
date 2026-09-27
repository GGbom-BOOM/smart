#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人员重识别(ReID)引擎 —— 判断画面里的人是不是"社区人员"。

流水线:
    YOLOv8 检测 person 框
      -> 裁剪 + resize 到 256x128 + ImageNet 归一化
      -> OSNet 提取 512 维特征 (可选水平翻转 TTA)
      -> L2 归一化后与参考图库做余弦相似度
      -> 最高相似度 >= 阈值 => 社区人员(带姓名), 否则 => 陌生人

本模块不依赖 ROS, 可以单独用脚本调用(见 build_gallery.py / 离线测试),
ROS 节点 reid_node.py 只是它的一层封装。
"""

import json
import os

import cv2
import numpy as np
import torch

from osnet import load_pretrained

# OSNet 训练时的标准输入尺寸 (高, 宽)
INPUT_H, INPUT_W = 256, 128
# ImageNet 归一化参数(OSNet 训练时用的就是这两个)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "..", "models", "osnet_x1_0_msmt17.pth")
DEFAULT_DETECTOR = os.path.join(_HERE, "..", "models", "yolov8n.pt")


def resolve_weights(name_or_path, kind="osnet"):
    """把权重参数解析成一个真实存在的路径。

    ultralytics 只会在**当前工作目录**找 yolov8n.pt, 找不到就去联网下载,
    而国内访问 GitHub assets 经常失败。所以在多个常见位置里找一遍,
    找到就用, 都找不到再交给 ultralytics 去下载(并给出明确提示)。
    """
    if os.path.isabs(name_or_path) or os.sep in name_or_path:
        if os.path.exists(name_or_path):
            return name_or_path
    candidates = [
        name_or_path,
        os.path.join(_HERE, name_or_path),                      # 脚本同级
        os.path.join(_HERE, "..", "models", name_or_path),      # 包的 models/
        os.path.join(os.path.expanduser("~"), ".ros", name_or_path),
        os.path.join(os.path.expanduser("~"), name_or_path),
        os.path.expanduser(os.path.join("~", "models", name_or_path)),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return name_or_path   # 都没找到: 原样返回, 由调用方决定是下载还是报错


# --------------------------------------------------------------------------
# 图像预处理
# --------------------------------------------------------------------------
def preprocess_crop(bgr_crop):
    """把一张 BGR 裁剪图变成 OSNet 的输入张量 (3, 256, 128)。

    注意 cv2.resize 的参数顺序是 (宽, 高), 和 torch 的 (高, 宽) 相反;
    这里不做等比缩放, 直接拉伸到 256x128 —— 与 torchreid 测试时的
    transforms.Resize([256,128]) 行为一致, 必须保持一致否则特征会偏移。
    """
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def _l2norm(v):
    """按行做 L2 归一化。余弦相似度 = 归一化后点积。"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# --------------------------------------------------------------------------
# 画框 + 中文标签
# --------------------------------------------------------------------------
# cv2.putText 画不了中文(会变成问号), 所以中文一律走 PIL。
CN_FONTS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
_FONT_CACHE = {}


def load_cn_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for p in CN_FONTS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                continue
    _FONT_CACHE[size] = None
    return None


def draw_detections(frame, results, show_topk=False):
    """在画面上画检测框和中文标签, 返回新图。

    绿框 = 社区人员, 红框 = 陌生人, 灰框 = 图库为空无法判断。
    整帧只做一次 BGR<->RGB 转换和一次 PIL 往返, 有多个框时也不会明显变慢。
    """
    ann = frame.copy()
    items = []
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        if r["is_member"]:
            color = (0, 200, 0)
            label = " %s  %.2f" % (r["name"], r["sim"])
        elif r["sim"] is None:
            color = (150, 150, 150)
            label = "未知(图库为空)"
        else:
            color = (0, 0, 255)
            label = "陌生人  %.2f" % r["sim"]
        if show_topk and r["topk"]:
            label += "  " + " ".join("%s:%.2f" % (n, s) for n, s in r["topk"][:3])
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        items.append((x1, y1, label, color))

    if not items:
        return ann
    font = load_cn_font(20)
    if font is None:
        # 没有中文字体时退回英文, 至少不要画出乱码
        for x1, y1, label, color in items:
            cv2.putText(ann, label.encode("ascii", "replace").decode(),
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return ann

    from PIL import Image as PILImage, ImageDraw
    pil = PILImage.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x1, y1, label, color in items:
        rgb = (color[2], color[1], color[0])
        # 文字底色描边, 避免浅色背景上看不清
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            d.text((x1 + dx, max(0, y1 - 24) + dy), label, font=font, fill=(0, 0, 0))
        d.text((x1, max(0, y1 - 24)), label, font=font, fill=rgb)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------
class ReIDEngine(object):
    def __init__(self, weights=DEFAULT_WEIGHTS, arch="osnet_x1_0",
                 detector_weights=DEFAULT_DETECTOR, device="cpu",
                 conf_det=0.5, min_person_h=40, tta_flip=True,
                 batch_size=8, threads=None):
        """
        weights       : OSNet 权重路径
        detector_weights: YOLOv8 权重(person 检测)
        conf_det      : 人框置信度阈值
        min_person_h  : 人框最小高度(像素), 太小的框看不清外观, 直接跳过
        tta_flip      : 是否用水平翻转做特征增强(精度更好, 耗时约x2)
        threads       : 限制 CPU 线程数。CPU 上跑时设成物理核数可以避免抢线程
        """
        if threads:
            torch.set_num_threads(int(threads))

        self.device = device
        self.conf_det = float(conf_det)
        self.min_person_h = int(min_person_h)
        self.tta_flip = bool(tta_flip)
        self.batch_size = int(batch_size)

        # --- 人检测 ---
        from ultralytics import YOLO
        det_path = resolve_weights(detector_weights, kind="detector")
        if not os.path.exists(det_path):
            raise FileNotFoundError(
                "找不到检测权重 %r。请把它放到 %s, 或用 detector_weights 参数指定绝对路径。"
                "(国内网络通常无法自动从 GitHub 下载)"
                % (detector_weights, os.path.normpath(DEFAULT_DETECTOR)))
        self.detector = YOLO(det_path)
        self.detector_path = det_path
        weights = resolve_weights(weights, kind="osnet")
        if not os.path.exists(weights):
            raise FileNotFoundError("找不到 OSNet 权重: %s" % weights)

        # --- 特征提取 ---
        self.model, info = load_pretrained(weights, arch=arch, device=device)
        self.weights_path = weights
        self.arch = arch
        self.num_classes = info["num_classes"]
        self.feature_dim = 512

        # --- 图库 ---
        self.gallery_feats = np.zeros((0, self.feature_dim), dtype=np.float32)
        self.gallery_names = []

    # ---------------- 特征 ----------------
    def extract(self, crops):
        """crops: list of BGR ndarray -> (N, 512) 已 L2 归一化的特征"""
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        feats = []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                chunk = crops[i:i + self.batch_size]
                x = torch.stack([preprocess_crop(c) for c in chunk]).to(self.device)
                f = self.model(x)
                if self.tta_flip:
                    # 原图和水平镜像各提一次, L2 归一化后取平均再归一化。
                    # 翻转不变性对 ReID 提升明显, 代价是多一倍前向。
                    f = f + self.model(torch.flip(x, dims=[3]))
                feats.append(f.cpu().numpy())
        return _l2norm(np.concatenate(feats, axis=0).astype(np.float32))

    # ---------------- 检测 ----------------
    def detect_persons(self, frame):
        """返回 [(x1,y1,x2,y2,conf), ...], 已按人框大小/置信度过滤"""
        res = self.detector(frame, verbose=False,
                            conf=self.conf_det, classes=[0])[0]  # COCO 0 = person
        boxes = []
        h_img = frame.shape[0]
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            if (y2 - y1) < self.min_person_h:
                continue
            # 裁到画面内(检测框可能略微越界)
            x1 = max(0.0, min(x1, frame.shape[1] - 1))
            x2 = max(0.0, min(x2, frame.shape[1]))
            y1 = max(0.0, min(y1, h_img - 1))
            y2 = max(0.0, min(y2, h_img))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        boxes.sort(key=lambda t: t[4], reverse=True)
        return boxes

    @staticmethod
    def crop(frame, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        return frame[y1:y2, x1:x2]

    # ---------------- 图库 ----------------
    def set_gallery(self, feats, names):
        """feats: (N,512) 未归一化或已归一化均可; names: 长度 N 的人名列表"""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.size == 0:
            self.gallery_feats = np.zeros((0, self.feature_dim), dtype=np.float32)
            self.gallery_names = []
            return
        self.gallery_feats = _l2norm(feats)
        self.gallery_names = list(names)

    def load_gallery(self, path):
        """从 build_gallery.py 生成的 .npz 文件加载图库"""
        d = np.load(path, allow_pickle=True)
        self.set_gallery(d["feats"], [str(s) for s in d["names"]])
        return len(self.gallery_names)

    def save_gallery(self, path, feats, names, sources=None, extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "feats": np.asarray(feats, dtype=np.float32),
            "names": np.array([str(n) for n in names]),
            "sources": np.array([str(s) for s in (sources or [""] * len(names))]),
            "meta": json.dumps(dict({"arch": self.arch,
                                     "model": os.path.basename(self.weights_path),
                                     "input": "%dx%d" % (INPUT_H, INPUT_W),
                                     "tta_flip": self.tta_flip}, **(extra or {}))),
        }
        np.savez_compressed(path, **payload)
        return path

    # ---------------- 主流程 ----------------
    def identify(self, frame, threshold=0.5):
        """对一帧做检测+识别。

        返回 list[dict], 每项:
            box         : (x1,y1,x2,y2)
            det_conf    : YOLO 的人框置信度
            is_member   : 是否判定为社区人员
            name        : 匹配到的姓名(非社区人员时为 None)
            sim         : 与图库的最高余弦相似度(图库为空时为 None)
            topk        : [(姓名, 相似度), ...] 相似度最高的前几个, 便于排查
        """
        boxes = self.detect_persons(frame)
        crops = [self.crop(frame, b) for b in boxes]
        crops = [c for c in crops if c.size > 0]
        if not crops:
            return []
        feats = self.extract(crops)          # (N,512) 已归一化

        gallery_empty = len(self.gallery_names) == 0
        out = []
        for box, feat in zip(boxes, feats):
            item = {"box": tuple(box[:4]), "det_conf": box[4],
                    "is_member": False, "name": None, "sim": None, "topk": []}
            if not gallery_empty:
                # 余弦相似度 = 归一化特征点积
                sims = self.gallery_feats.dot(feat)       # (N,)
                best_i = int(np.argmax(sims))
                item["sim"] = float(sims[best_i])
                item["name"] = self.gallery_names[best_i]
                item["is_member"] = bool(sims[best_i] >= threshold)
                if not item["is_member"]:
                    item["name"] = None
                # top-k(按人聚合取最高, 避免同一个人多张图霸榜)
                per_person = {}
                for s, n in zip(sims, self.gallery_names):
                    if n not in per_person or s > per_person[n]:
                        per_person[n] = float(s)
                item["topk"] = sorted(per_person.items(),
                                      key=lambda kv: kv[1], reverse=True)[:5]
            out.append(item)
        return out


# --------------------------------------------------------------------------
# 离线自测
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="ReID 引擎离线自测")
    ap.add_argument("image", help="测试图片路径")
    ap.add_argument("--gallery", default="", help="图库 .npz 路径(可选)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-tta", action="store_true", help="关闭翻转增强")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default="", help="把标注结果存到该路径")
    args = ap.parse_args()

    eng = ReIDEngine(weights=args.weights, tta_flip=not args.no_tta,
                     threads=args.threads or None)
    if args.gallery:
        n = eng.load_gallery(args.gallery)
        print("图库载入: %d 条特征" % n)

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit("读不到图片: %s" % args.image)

    t0 = time.time()
    results = eng.identify(frame, threshold=args.threshold)
    dt = time.time() - t0
    print("检测到 %d 个人, 耗时 %.3f s" % (len(results), dt))
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        tag = ("社区人员:%s" % r["name"]) if r["is_member"] else "陌生人"
        print("  框(%d,%d,%d,%d) 检测%.2f -> %s sim=%s"
              % (x1, y1, x2, y2, r["det_conf"], tag,
                 "%.3f" % r["sim"] if r["sim"] is not None else "N/A"))
        for n, s in r["topk"]:
            print("      %-12s %.3f" % (n, s))

    if args.save:
        cv2.imwrite(args.save, draw_detections(frame, results))
        print("标注结果已存到", args.save)
