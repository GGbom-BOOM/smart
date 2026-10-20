#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相似度阈值标定 —— 决定"多像才算社区人员"。

这是整个系统最容易被忽略、却最影响实际效果的一步。余弦相似度的绝对值
在不同场景(仿真/实拍、光照、服装、模型)下差别很大, 拍脑袋定 0.5 或 0.8
都不靠谱, 必须用你自己的数据扫一遍。

准备两份**独立于图库**的数据:
  正样本 pos/: 社区人员的**其它**照片(不要用建库那几张), 按姓名分子目录
        pos/张三/*.jpg
  负样本 neg/: 非社区人员(路人、陌生人)的照片, 平铺即可
        neg/*.jpg

用法:
    python3 calibrate_threshold.py --gallery ./gallery/gallery.npz \
        --pos ./calib/pos --neg ./calib/neg

输出: 各阈值下的"社区人员通过率(TPR)"和"陌生人误放行率(FPR)",
以及推荐的阈值(最大化 TPR-FPR)。强烈建议正负样本各准备 30 张以上。
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_engine import ReIDEngine, DEFAULT_WEIGHTS  # noqa: E402
from build_gallery import list_images                # noqa: E402


def collect_samples(engine, root, is_positive):
    """返回 [(sim, correct_name_or_None, 路径), ...]

    正样本: 按子目录名 = 真实姓名; 只有匹配对了才算"通过"
    负样本: 匹配到任何人都是"误放行"
    """
    samples = []
    if is_positive:
        for person in sorted(os.listdir(root)):
            pdir = os.path.join(root, person)
            if not os.path.isdir(pdir):
                continue
            for p in list_images(pdir):
                samples.append((p, person))
    else:
        for p in list_images(root):
            samples.append((p, None))

    out = []
    for p, truth in samples:
        img = cv2.imread(p)
        if img is None:
            print("  [跳过] 读不到 %s" % p)
            continue
        boxes = engine.detect_persons(img)
        if not boxes:
            # 没检出人就退回整图, 保证标定仍能进行
            boxes = [(0, 0, img.shape[1], img.shape[0], 1.0)]
        # 取面积最大的人
        b = max(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
        crop = engine.crop(img, b)
        if crop.size == 0:
            continue
        feat = engine.extract([crop])[0]
        if len(engine.gallery_names) == 0:
            raise SystemExit("图库是空的, 请先跑 build_gallery.py")
        sims = engine.gallery_feats.dot(feat)
        bi = int(np.argmax(sims))
        top1_name, top1_sim = engine.gallery_names[bi], float(sims[bi])
        correct = (top1_name == truth) if truth is not None else None
        out.append((top1_sim, correct, p))
    return out


def main():
    ap = argparse.ArgumentParser(description="标定社区人员判定阈值")
    ap.add_argument("--gallery", required=True, help="图库 .npz")
    ap.add_argument("--pos", required=True, help="正样本目录(子目录名=姓名)")
    ap.add_argument("--neg", required=True, help="负样本目录(陌生人照片)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    eng = ReIDEngine(weights=args.weights, tta_flip=not args.no_tta,
                     threads=args.threads or None)
    n = eng.load_gallery(args.gallery)
    print("图库: %d 条特征, %d 个人\n" % (n, len(set(eng.gallery_names))))

    print("提取正样本(社区人员)...")
    pos = collect_samples(eng, args.pos, True)
    print("  有效 %d 张" % len(pos))
    print("提取负样本(陌生人)...")
    neg = collect_samples(eng, args.neg, False)
    print("  有效 %d 张\n" % len(neg))

    if not pos or not neg:
        raise SystemExit("正/负样本不能为空")

    pos_sims = np.array([s for s, _, _ in pos])
    neg_sims = np.array([s for s, _, _ in neg])

    print("相似度分布:")
    print("  社区人员 : 均值 %.3f  最低 %.3f  最高 %.3f"
          % (pos_sims.mean(), pos_sims.min(), pos_sims.max()))
    print("  陌生人   : 均值 %.3f  最低 %.3f  最高 %.3f"
          % (neg_sims.mean(), neg_sims.min(), neg_sims.max()))

    # 先把"匹配对了但相似度不够"和"匹配错了"分开统计, 方便定位问题
    wrong_id = [p for s, c, p in pos if c is False]
    if wrong_id:
        print("\n  注意: 有 %d 张正样本 top-1 匹配到了错误的人, 例如:" % len(wrong_id))
        for p in wrong_id[:3]:
            print("    %s" % p)

    print("\n阈值扫描:")
    print("  阈值   社区人员通过率   陌生人误放行率   TPR-FPR")
    best = (None, -9)
    rows = []
    for t in np.arange(0.30, 0.96, 0.02):
        tpr = float((pos_sims >= t).mean())
        fpr = float((neg_sims >= t).mean())
        rows.append((t, tpr, fpr, tpr - fpr))
        if tpr - fpr > best[1]:
            best = (t, tpr - fpr)
    for t, tpr, fpr, j in rows:
        mark = "  <== 推荐" if abs(t - best[0]) < 1e-9 else ""
        print("  %.2f      %6.1f%%          %6.1f%%       %+.3f%s"
              % (t, tpr * 100, fpr * 100, j, mark))

    t = best[0]
    tpr = float((pos_sims >= t).mean())
    fpr = float((neg_sims >= t).mean())
    print("\n推荐阈值: %.2f" % t)
    print("  社区人员通过率 %.1f%% (漏拒 %.1f%%)" % (tpr * 100, (1 - tpr) * 100))
    print("  陌生人误放行率 %.1f%%" % (fpr * 100))

    # 给一个更保守的选择: 误放行控制在 5% 以内时最高的通过率
    safe = [(tt, tp, fp) for tt, tp, fp, _ in rows if fp <= 0.05]
    if safe:
        st, stp, sfp = max(safe, key=lambda r: r[1])
        print("\n若更看重「不误放行」(误放行<=5%%), 可用阈值 %.2f: 通过率 %.1f%%, 误放行 %.1f%%"
              % (st, stp * 100, sfp * 100))
    else:
        print("\n注意: 没有任何阈值能把误放行压到 5%% 以下, "
              "说明陌生人和社区人员的特征区分度不足。")
        print("      建议: 补充参考图 / 提高相机分辨率 / 换更贴合场景的 ReID 模型。")

    mid = (pos_sims.min() + neg_sims.max()) / 2.0
    if pos_sims.min() > neg_sims.max():
        print("\n两类分布完全不重叠(间隔 %.3f), 系统在该数据上可分性很好。" %
              (pos_sims.min() - neg_sims.max()))
    else:
        print("\n两类分布有重叠(重叠区间约 %.3f ~ %.3f), 无法做到零错误, "
              "阈值是在「漏拒」和「误放行」之间取舍。"
              % (neg_sims.max(), pos_sims.min()))


if __name__ == "__main__":
    main()
