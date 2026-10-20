#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键自检 —— 确认 person_reid 装好了、能跑、且精度正常。

    python3 scripts/selftest.py

不需要 ROS, 不需要仿真, 不需要相机。用 ultralytics 自带的 bus.jpg / zidane.jpg
(含真实行人)跑完整流程, 每一关打印 通过/失败。

退出码 0 = 全过; 非 0 = 有失败项。
"""

import os
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, FAIL, SKIP = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m-\033[0m"
results = []


def stage(name):
    print("\n[%s]" % name)


def check(desc, ok, detail=""):
    results.append((desc, ok))
    mark = OK if ok else FAIL
    print("  %s %s%s" % (mark, desc, ("  " + detail) if detail else ""))
    return ok


def skip(desc, why):
    print("  %s %s  (%s)" % (SKIP, desc, why))


def main():
    print("=" * 62)
    print("person_reid 自检")
    print("=" * 62)

    # ---------- 1. 依赖 ----------
    stage("1/6 依赖")
    missing = []
    for mod in ("torch", "torchvision", "cv2", "numpy", "ultralytics", "PIL"):
        try:
            __import__(mod)
            check("import %s" % mod, True)
        except ImportError as e:
            missing.append(mod)
            check("import %s" % mod, False, str(e))
    if missing:
        print("\n先装依赖: pip3 install " + " ".join(missing))
        return 1

    import cv2
    import torch
    check("CUDA 可用性", True,
          "GPU 可用" if torch.cuda.is_available() else
          "无 GPU, 用 CPU(正常, 只是慢)")

    # 中文字体(标注图要显示姓名)
    from reid_engine import load_cn_font
    check("中文字体", load_cn_font(20) is not None,
          "" if load_cn_font(20) else "找不到 CJK 字体, 标注图会退回英文")

    # ---------- 2. 权重 ----------
    stage("2/6 权重")
    from reid_engine import ReIDEngine, DEFAULT_WEIGHTS, DEFAULT_DETECTOR, resolve_weights
    w = resolve_weights(DEFAULT_WEIGHTS)
    d = resolve_weights(DEFAULT_DETECTOR)
    check("OSNet 权重", os.path.exists(w), w)
    check("YOLOv8 权重", os.path.exists(d), d)
    if not (os.path.exists(w) and os.path.exists(d)):
        print("\n权重缺失, 下载方法见 README「二、下载权重」")
        return 1

    # ---------- 3. 网络结构 ----------
    stage("3/6 OSNet 结构")
    try:
        from osnet import osnet_x1_0
        sd = torch.load(w, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and \
                not any(k.startswith("conv1.") for k in sd):
            sd = sd["state_dict"]
        m = osnet_x1_0(num_classes=int(sd["classifier.weight"].shape[0]))
        m.load_state_dict(sd, strict=True)   # 一个 key 都不能差
        m.eval()
        with torch.no_grad():
            f = m(torch.randn(1, 3, 256, 128))
        check("官方权重 strict=True 加载", True, "分类头 %d 类" % m.classifier.out_features)
        check("前向输出 512 维", tuple(f.shape) == (1, 512), str(tuple(f.shape)))
    except Exception as e:
        check("OSNet 结构校验", False, repr(e))
        traceback.print_exc()
        return 1

    # ---------- 4. 检测 ----------
    stage("4/6 人检测")
    import ultralytics
    assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    bus = os.path.join(assets, "bus.jpg")
    zid = os.path.join(assets, "zidane.jpg")
    if not (os.path.exists(bus) and os.path.exists(zid)):
        skip("YOLO 检测测试", "找不到 ultralytics 自带测试图")
        print("\n无法继续。可以从别处找一张含人的图, 用")
        print("  python3 scripts/reid_engine.py 你的图.jpg --save /tmp/out.jpg")
        print("手动验证。")
        return 0

    t0 = time.time()
    eng = ReIDEngine(tta_flip=True, threads=2)
    check("引擎初始化", True, "%.1fs" % (time.time() - t0))

    def crops(path):
        fr = cv2.imread(path)
        return [eng.crop(fr, b) for b in eng.detect_persons(fr)]

    bus_c, zid_c = crops(bus), crops(zid)
    check("bus.jpg 检出人数", len(bus_c) >= 3, "检出 %d 人(期望 >=3)" % len(bus_c))
    check("zidane.jpg 检出人数", len(zid_c) >= 2, "检出 %d 人(期望 >=2)" % len(zid_c))
    if len(bus_c) < 2 or len(zid_c) < 1:
        check("检出数量足够", False, "太少, 换测试图")
        return 1

    # ---------- 5. 特征判别力 ----------
    stage("5/6 特征判别力(核心)")
    people = bus_c[:2] + zid_c[:1]
    names = ["bus#0", "bus#1", "zidane#0"]
    base = eng.extract(people)

    def aug(c, k):
        if k == 1:
            return np.clip(c.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        if k == 2:
            return np.clip(c.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
        if k == 3:
            ok, e = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 30])
            return cv2.imdecode(e, 1)
        return cv2.flip(c, 1)

    same = []
    for i, c in enumerate(people):
        a = eng.extract([aug(c, k) for k in range(1, 5)])
        same += [float(base[i] @ a[k]) for k in range(4)]
    cross = [float(base[i] @ base[j])
             for i in range(len(people)) for j in range(i + 1, len(people))]

    print("    同人(原图 vs 变体) 最低 %.3f  均值 %.3f" % (np.min(same), np.mean(same)))
    print("    异人              最高 %.3f  均值 %.3f" % (np.max(cross), np.mean(cross)))
    gap = float(np.min(same) - np.max(cross))
    check("同人相似度 > 0.8", float(np.min(same)) > 0.8, "最低 %.3f" % np.min(same))
    check("异人相似度 < 0.7", float(np.max(cross)) < 0.7, "最高 %.3f" % np.max(cross))
    check("两类可分(间隔 > 0.15)", gap > 0.15, "间隔 %.3f" % gap)

    # ---------- 6. 全流程 ----------
    stage("6/6 全流程(建库 -> 标定 -> 识别)")
    try:
        tmp = tempfile.mkdtemp(prefix="reid_selftest_")
        # 图库: bus#0 叫"甲", zidane#0 叫"乙"
        for nm, c in (("甲", bus_c[0]), ("乙", zid_c[0])):
            os.makedirs(os.path.join(tmp, "gal", nm))
            cv2.imwrite(os.path.join(tmp, "gal", nm, "ref.jpg"), c)
        # 正样本: 甲的变体; 负样本: 不相干的人
        os.makedirs(os.path.join(tmp, "pos", "甲"))
        for k in (1, 2, 3, 4):
            cv2.imwrite(os.path.join(tmp, "pos", "甲", "v%d.jpg" % k), aug(bus_c[0], k))
        os.makedirs(os.path.join(tmp, "neg"))
        for i, c in enumerate(bus_c[1:] + zid_c[1:]):
            cv2.imwrite(os.path.join(tmp, "neg", "n%d.jpg" % i), c)

        eng.save_gallery(os.path.join(tmp, "g.npz"), eng.extract(people[:2]),
                         ["甲", "乙"])
        n = eng.load_gallery(os.path.join(tmp, "g.npz"))
        check("建库/存盘/载入", n == 2, "%d 条特征" % n)

        pos = [float(eng.gallery_feats.dot(eng.extract([aug(bus_c[0], k)])[0]).max())
               for k in (1, 2, 3, 4)]
        neg = []
        for c in bus_c[1:] + zid_c[1:]:
            neg.append(float(eng.gallery_feats.dot(eng.extract([c])[0]).max()))
        thr = (np.min(pos) + np.max(neg)) / 2.0
        print("    正样本相似度 %s" % " ".join("%.3f" % v for v in pos))
        print("    负样本相似度 %s" % " ".join("%.3f" % v for v in neg))
        check("标定能得到有效阈值", np.min(pos) > np.max(neg), "阈值 %.3f" % thr)

        res = eng.identify(cv2.imread(bus), threshold=thr)
        hit = [r for r in res if r["is_member"] and r["name"] == "甲"]
        check("识别命中已知人员", len(hit) >= 1,
              "检出 %d 人, 命中甲 %d 次" % (len(res), len(hit)))

        out = draw = None
        from reid_engine import draw_detections
        draw = draw_detections(cv2.imread(bus), res)
        out = os.path.join(tmp, "annotated.jpg")
        cv2.imwrite(out, draw)
        check("标注图生成", os.path.exists(out), out)
        print("\n    标注图留在: %s" % out)
        print("    (想清掉就 rm -rf %s)" % tmp)
    except Exception as e:
        check("全流程", False, repr(e))
        traceback.print_exc()
        return 1

    # ---------- 汇总 ----------
    print("\n" + "=" * 62)
    bad = [d for d, ok in results if not ok]
    if bad:
        print("结果: %d 项失败" % len(bad))
        for d in bad:
            print("  %s %s" % (FAIL, d))
        return 1
    print("结果: 全部 %d 项通过 —— 系统可以用了" % len(results))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
