"""นับเม็ดยาด้วยวิธีคลาสสิก 3 วิธี (CC / หารพื้นที่ / watershed) ด้วย pipeline และค่าเริ่มต้นเดียวกับแอป web/index.html

ใช้แบบ batch:
    python pillcount.py images/ truth.csv -o results.csv [--chan s --wsf 0.5 ...]
truth.csv มีคอลัมน์ file,truth,density,background (density/background ใช้จัดกลุ่ม MAPE)
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ค่าเริ่มต้นตรงกับแอป v1
DEFAULTS = dict(chan="gray", blur="gauss", ksize=5, thresh="otsu", thv=128, block=51, c=-5,
                inv="auto", mk=3, open=2, close=1, fill=1, mina=0.25, wsf=0.60, maxd=900)


def to_channel(rgb, mode, bg):
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode in ("r", "g", "b"):
        v = rgb[..., "rgb".index(mode)]
    elif mode == "s":
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        v = np.where(mx == 0, 0, (mx - mn) * 255 / np.maximum(mx, 1))
    elif mode == "v":
        v = rgb.max(axis=2)
    elif mode == "bg":
        v = np.minimum(255, np.sqrt(((rgb - np.float32(bg)) ** 2).sum(axis=2)) * 1.4)
    else:
        v = 0.299 * r + 0.587 * g + 0.114 * b
    return v.astype(np.uint8)  # ตัดทศนิยมทิ้งแบบ Uint8Array ใน JS


def border_color(rgb):
    edge = np.concatenate([rgb[0], rgb[-1], rgb[2], rgb[-3], rgb[:, 0], rgb[:, -1], rgb[:, 2], rgb[:, -3]])
    return np.floor(edge.mean(axis=0) + 0.5).astype(int).tolist()


def count(rgb, p=DEFAULTS, bg=None, ref_area=None):
    """rgb: ภาพ uint8 (H,W,3) ที่ย่อแล้ว คืน dict ผลนับ 3 วิธี + ภาพกลางทาง"""
    p = {**DEFAULTS, **p}
    bg = bg or border_color(rgb)
    chan = to_channel(rgb, p["chan"], bg)
    k = int(p["ksize"])
    if p["blur"] == "mean":
        blur = cv2.blur(chan, (k, k), borderType=cv2.BORDER_REPLICATE)
    elif p["blur"] == "gauss":
        blur = cv2.GaussianBlur(chan, (k, k), 0, borderType=cv2.BORDER_REPLICATE)
    elif p["blur"] == "median":
        blur = cv2.medianBlur(chan, k)
    else:
        blur = chan

    th = None
    if p["thresh"] == "otsu":
        th, bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif p["thresh"] == "manual":
        th, bin_ = cv2.threshold(blur, int(p["thv"]), 255, cv2.THRESH_BINARY)
    else:
        method = cv2.ADAPTIVE_THRESH_MEAN_C if p["thresh"] == "amean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        bin_ = cv2.adaptiveThreshold(blur, 255, method, cv2.THRESH_BINARY, int(p["block"]), float(p["c"]))

    # ขอบภาพเป็นขาวเกินครึ่ง = พื้นหลังขาว → กลับให้เม็ดเป็นขาว
    invert = p["inv"] == "inv"
    if p["inv"] == "auto":
        edge = np.concatenate([bin_[0], bin_[-1], bin_[:, 0], bin_[:, -1]])
        invert = (edge > 0).mean() > 0.5
    if invert:
        bin_ = 255 - bin_

    kern = np.ones((int(p["mk"]), int(p["mk"])), np.uint8)
    m = bin_
    if p["open"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern, iterations=int(p["open"]))
    if p["close"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=int(p["close"]))
    if p["fill"]:  # รูที่ไม่ต่อกับขอบภาพ = รูในเม็ด
        n, lab = cv2.connectedComponents((m == 0).astype(np.uint8), connectivity=4)
        outside = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        m = np.where((m > 0) | ~np.isin(lab, outside), 255, 0).astype(np.uint8)

    # Method A: connected components + ตัดก้อนเล็ก
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas[areas >= 20]
    med_a = float(np.median(big)) if len(big) else 0.0
    min_area = max(12, med_a * float(p["mina"]))
    keep = np.zeros(n, bool)
    keep[1:] = areas >= min_area
    mask = keep[lab]
    kept_areas = areas[keep[1:]]

    # Method C: distance transform + marker watershed (marker = จุดที่ dist ≥ wsf × dist สูงสุดของก้อนนั้น)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cmax = np.zeros(n, np.float32)
    np.maximum.at(cmax, lab[mask], dist[mask])
    peaks = mask & (dist >= float(p["wsf"]) * cmax[lab]) & (dist >= 1)
    n_mk, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    ws_count = n_mk - 1
    markers = markers.astype(np.int32)
    # cv2.watershed ไหลตามความต่างสีของภาพจริง (สูตรมาตรฐาน OpenCV) ต่างจากแอปที่ไหลตาม dist
    # จำนวน (= จำนวน marker) จึงเท่ากัน แต่พื้นที่ region ที่ใช้เป็น ref ของวิธีหารพื้นที่อาจต่างกัน
    # พื้นหลังเป็น label แยก เว้นวงรอบเม็ด 2 px ไม่ให้พื้นหลังแย่งขอบเม็ด
    markers[cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0] = n_mk
    cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)
    ws_areas = np.bincount(markers[mask & (markers > 0)].ravel(), minlength=n_mk + 1)[1:n_mk]

    # Method B: หารพื้นที่ ref = มัธยฐานพื้นที่ region ของ watershed (≈ 1 เม็ด) หรือเม็ดที่กำหนดเอง
    # shortcut: ยังไม่มี guard ref เล็กผิดปกติ (เคส 5436 ใน HANDOFF) เหมือนแอป v1 เพื่อให้ผลเทียบกันได้
    ref = ref_area or (float(np.median(ws_areas[ws_areas > 0])) if (ws_areas > 0).any() else 0) \
        or (float(np.median(kept_areas)) if len(kept_areas) else 0) or 1.0
    area_total = int(sum(max(1, int(np.floor(a / ref + 0.5))) for a in kept_areas))

    return dict(cc=int(keep.sum()), area=area_total, ws=ws_count, ref=ref, th=th, invert=bool(invert),
                bg=bg, mask=mask, dist=dist, markers=markers)


def load(path, maxd=DEFAULTS["maxd"]):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"เปิดภาพไม่ได้: {path} (iPhone: ส่งออกเป็น JPG ก่อน ไม่ใช่ HEIC)")
    s = min(1.0, maxd / max(bgr.shape[:2]))
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mape(rows, key):
    errs = [abs(r[key] - r["truth"]) / r["truth"] * 100 for r in rows if r["truth"] > 0]
    return round(statistics.mean(errs), 2) if errs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results.csv"))
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args(argv)
    p = {k: getattr(a, k) for k in DEFAULTS}

    with open(a.truth, newline="", encoding="utf-8-sig") as f:
        truth = list(csv.DictReader(f))
    rows = []
    for t in truth:
        t0 = time.perf_counter()
        r = count(load(a.images / t["file"], p["maxd"]), p)
        rows.append(dict(file=t["file"], density=t.get("density", ""), background=t.get("background", ""),
                         truth=int(t["truth"]), cc=r["cc"], area=r["area"], ws=r["ws"], ref=round(r["ref"], 1),
                         ms=round((time.perf_counter() - t0) * 1000),
                         params=" ".join(f"{k}={v}" for k, v in p.items() if v != DEFAULTS[k])))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault((r["density"], r["background"]), []).append(r)
    print("density,background,n,MAPE_cc,MAPE_area,MAPE_ws")
    for (d, b), g in sorted(groups.items()):
        print(f"{d},{b},{len(g)},{mape(g, 'cc')},{mape(g, 'area')},{mape(g, 'ws')}")
    print(f"all,all,{len(rows)},{mape(rows, 'cc')},{mape(rows, 'area')},{mape(rows, 'ws')}")
    print(f"→ {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
