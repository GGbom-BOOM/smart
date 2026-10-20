#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从参考图片目录构建 ReID 特征库。

两种组织方式, 可以混用 —— 人名就是身份标签:

  方式A: 一个子目录一个人(推荐, 人多了好管理)
      gallery/
      ├── 张三/
      │   ├── 正面.jpg
      │   └── 侧面.jpg
      └── 李四/
          └── 1.jpg

  方式B: 平铺, 文件名用 '_' 带人名前缀(不用建文件夹)
      gallery/
      ├── 张三_正面.jpg
      ├── 张三_侧面.jpg
      └── 李四_1.jpg

  一个人只有一张图时, 可以直接叫 gallery/张三.jpg (文件名主体当人名)。

用法:
    python3 build_gallery.py --src ../gallery --out ../gallery/gallery.npz
    python3 build_gallery.py --src ../gallery --out ../gallery/gallery.npz --sep _

参考图既可以是"已经裁好的人像", 也可以是"含人的场景照" —— 默认会先用
YOLOv8 找到人并自动裁剪, 找不到人时退回用整张图(--no-auto-crop 可强制用整图)。

建议每个社区人员准备 3~8 张不同角度/光照/距离的图。图越多越稳, 但边际收益递减。
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_engine import ReIDEngine, DEFAULT_WEIGHTS  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


def collect_person_images(src, sep="_"):
    """收集 {人名: [图片路径]}。支持两种组织方式, 可以混用:

      A. 子目录名 = 人名(推荐, 人多了好管理)
             gallery/张三/正面.jpg   gallery/张三/侧面.jpg

      B. 平铺, 文件名用分隔符带人名前缀(不用建文件夹)
             gallery/张三_正面.jpg   gallery/张三_侧面.jpg   gallery/李四_1.jpg

    文件名里没有分隔符时, 直接用文件名主体当人名 —— 所以
    一个人一张图的情况可以叫 gallery/张三.jpg, 也能正常工作。
    """
    groups = {}
    # A. 每个子目录 = 一个人
    for entry in sorted(os.listdir(src)):
        pdir = os.path.join(src, entry)
        if not os.path.isdir(pdir) or entry.startswith("."):
            continue
        imgs = list_images(pdir)
        if imgs:
            groups.setdefault(entry, []).extend(imgs)
    # B. 直接躺在 src 下的图片(注意 list_images 是递归的, 这里只取第一层)
    for p in list_images(src):
        if os.path.dirname(p) != src:
            continue
        base = os.path.basename(p)
        if sep and sep in base:
            person = base.split(sep)[0]
        else:
            person = os.path.splitext(base)[0]
        groups.setdefault(person, []).append(p)
    return groups


def pick_person_crop(engine, img, auto_crop=True):
    """从参考图里取出"人"的裁剪。

    auto_crop=True 时优先用 YOLO 检出的最大人框; 没检出人(比如图已经是
    裁好的头像/半身像且 YOLO 漏检)就退回整张图。
    """
    if auto_crop:
        boxes = engine.detect_persons(img)
        if boxes:
            # 取面积最大的那个人
            b = max(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
            c = engine.crop(img, b)
            if c.size > 0:
                return c, "yolo"
    return img, "full"


def main():
    ap = argparse.ArgumentParser(description="构建 ReID 参考特征库")
    ap.add_argument("--src", required=True, help="参考图根目录(子目录名=姓名)")
    ap.add_argument("--out", default="gallery.npz", help="输出 .npz 路径")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-auto-crop", action="store_true",
                    help="不做 YOLO 自动裁剪, 直接用整张参考图")
    ap.add_argument("--no-tta", action="store_true", help="关闭翻转增强")
    ap.add_argument("--sep", default="_",
                    help="平铺模式下文件名里的人名分隔符, 默认 '_' "
                         "(例如 张三_正面.jpg -> 人名'张三')")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-crops", default="",
                    help="把人像裁剪另存到该目录, 便于人工核对(可选)")
    args = ap.parse_args()

    auto_crop = not args.no_auto_crop
    # 建库时不开 TTA: 库里的特征越"干净"越稳定, 查询侧开 TTA 就足以吸收抖动。
    eng = ReIDEngine(weights=args.weights, tta_flip=not args.no_tta,
                     threads=args.threads or None)

    if not os.path.isdir(args.src):
        raise SystemExit("参考图目录不存在: %s" % args.src)

    groups = collect_person_images(args.src, sep=args.sep)
    if not groups:
        raise SystemExit("在 %s 里没找到任何图片" % args.src)

    feats, names, sources = [], [], []
    n_skip = 0
    for person in sorted(groups):
        crops, srcs = [], []
        for p in groups[person]:
            img = cv2.imread(p)
            if img is None:
                print("  [跳过] 读不到: %s" % p)
                n_skip += 1
                continue
            crop, how = pick_person_crop(eng, img, auto_crop)
            crops.append(crop)
            srcs.append(p)
            if args.save_crops:
                od = os.path.join(args.save_crops, person)
                os.makedirs(od, exist_ok=True)
                cv2.imwrite(os.path.join(od, os.path.basename(p)), crop)
        if not crops:
            continue
        f = eng.extract(crops)
        feats.append(f)
        names.extend([person] * len(crops))
        sources.extend(srcs)
        print("  %-16s %d 张参考图 -> %d 条特征" % (person, len(crops), len(crops)))

    if not feats:
        raise SystemExit("没有提取到任何特征, 请检查 %s 的目录结构" % args.src)

    feats = np.concatenate(feats, axis=0)
    eng.save_gallery(args.out, feats, names, sources)
    print("\n图库已保存: %s" % os.path.abspath(args.out))
    print("  人员 %d 人, 特征 %d 条, 维度 %d"
          % (len(set(names)), len(names), feats.shape[1]))

    # 自检: 同一个人内部的特征互相有多像? 太低说明参考图质量差/不是同一个人。
    print("\n图库自检(同一人内部相似度, 应明显高于阈值):")
    for person in sorted(set(names)):
        idx = [i for i, n in enumerate(names) if n == person]
        if len(idx) < 2:
            print("  %-16s 只有1张图, 无法自检" % person)
            continue
        sub = feats[idx]
        sims = [float(sub[a] @ sub[b])
                for a in range(len(idx)) for b in range(a + 1, len(idx))]
        flag = "  <-- 偏低, 建议换更好的参考图" if np.mean(sims) < 0.6 else ""
        print("  %-16s 均值 %.3f  最低 %.3f%s"
              % (person, np.mean(sims), np.min(sims), flag))

    if n_skip:
        print("\n共跳过 %d 张无法读取的图片" % n_skip)


if __name__ == "__main__":
    main()
