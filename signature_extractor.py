# -*- coding: utf-8 -*-
"""签名提取工具：从图片中提取手写签名，输出固定尺寸（默认 280x80）的透明/白底 PNG。"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUT_DIR_NAME = "签名提取"
OUT_SUFFIX = "_签名"
WORK_MAX_SIDE = 2500  # 处理时的最大边长，输出很小，没必要用全分辨率

INK_COLORS = {
    "黑色": (0, 0, 0),
    "原墨色": None,
    "深蓝": (20, 40, 140),
    "红色": (200, 20, 20),
}


# ---------------------------------------------------------------- 核心算法

def load_image(path):
    """读取图片为 RGB numpy 数组（支持中文路径、EXIF 旋转、透明通道）。"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.array(im.convert("RGB"))


# ---- 基础图像操作：只用 numpy + Pillow 实现，不依赖 OpenCV（打包体积小很多）

def _dilate(mask, size):
    """二值图 size x size 矩形膨胀，按行、列两次滑动窗口求和实现。"""
    r = size // 2

    def along_rows(m):
        h, w = m.shape
        cs = np.zeros((h, w + 1), np.int32)
        np.cumsum(m, axis=1, out=cs[:, 1:])
        x = np.arange(w)
        return (cs[:, np.minimum(x + r + 1, w)] - cs[:, np.maximum(x - r, 0)]) > 0

    # 只处理前景外框（外扩 r）这一块，签名通常只占图片一小部分
    out = np.zeros(mask.shape, bool)
    ys, xs = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if ys.size:
        y0, y1 = max(ys[0] - r, 0), ys[-1] + r + 1
        x0, x1 = max(xs[0] - r, 0), xs[-1] + r + 1
        out[y0:y1, x0:x1] = along_rows(along_rows(mask[y0:y1, x0:x1] > 0).T).T
    return out


def _otsu(gray):
    """Otsu 自动阈值。"""
    p = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.argmax(np.nan_to_num(sigma)))


def _components(mask):
    """8 连通域标记。返回 (标签图, 连通域数量含背景, stats[x, y, 宽, 高, 面积])。

    先把每行的连续前景段（游程）找出来，相邻两行重叠的游程即相连，
    再在游程图上做最小标签传播，全程向量化。
    """
    h, w = mask.shape
    pad = np.zeros((h, w + 2), np.int8)
    pad[:, 1:-1] = mask > 0
    d = np.diff(pad, axis=1)
    rows, starts = np.nonzero(d == 1)
    ends = np.nonzero(d == -1)[1]  # 不含
    n = len(starts)
    if n == 0:
        return np.zeros((h, w), np.int32), 1, np.zeros((1, 5), np.int64)

    # 第 r 行的游程 i 与第 r+1 行的游程 j 相连：s_j <= e_i 且 e_j >= s_i（8 连通）
    k = w + 2
    lo = np.searchsorted(rows * k + ends, (rows + 1) * k + starts, "left")
    hi = np.searchsorted(rows * k + starts, (rows + 1) * k + ends, "right")
    cnt = np.maximum(hi - lo, 0)
    a = np.repeat(np.arange(n), cnt)
    b = np.repeat(lo - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())

    lab = np.arange(n)
    while a.size:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        for idx in (a, b, lab[a], lab[b]):
            np.minimum.at(new, idx, m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new

    comp = np.unique(lab, return_inverse=True)[1].ravel() + 1  # 0 为背景
    nc = int(comp.max()) + 1
    x0 = np.full(nc, w)
    y0 = np.full(nc, h)
    x1 = np.zeros(nc, np.int64)
    y1 = np.zeros(nc, np.int64)
    np.minimum.at(x0, comp, starts)
    np.minimum.at(y0, comp, rows)
    np.maximum.at(x1, comp, ends)
    np.maximum.at(y1, comp, rows + 1)
    area = np.bincount(comp, weights=ends - starts, minlength=nc)
    stats = np.stack([x0, y0, x1 - x0, y1 - y0, area], axis=1).astype(np.int64)
    stats[0] = 0

    # 标签图：每段起点 +标签、终点 -标签，再按行累加
    acc = np.zeros((h, w + 1), np.int32)
    acc[rows, starts] += comp
    acc[rows, ends] -= comp
    return np.cumsum(acc, axis=1)[:, :w], nc, stats


def _normalize_background(gray):
    """去除光照不均/阴影：估计纸张背景后做除法归一化，纸张≈255，墨迹偏暗。"""
    h, w = gray.shape
    im = Image.fromarray(gray)
    scale = 400.0 / max(h, w)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BOX)
    k = max(7, (min(im.size) // 25) | 1)
    bg = im.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MedianFilter(k))
    bg = np.asarray(bg.resize((w, h), Image.BILINEAR), np.float32)
    norm = gray.astype(np.float32) * 255.0 / np.maximum(bg, 1.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def _select_signature(mask):
    """去噪点、去边框线，并把相邻笔画聚成一组，只保留最主要的签名区域。"""
    h, w = mask.shape
    labels, n, stats = _components(mask)
    min_area = max(6, int(h * w * 0.000015))
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        # 贴边且很长的多半是纸张边缘、扫描阴影、表格线
        if touches and (bw > 0.5 * w or bh > 0.5 * h):
            continue
        # 极细长的直线（表格线/下划线）
        if (bw > 0.3 * w and bh < 0.01 * h + 3) or (bh > 0.3 * h and bw < 0.01 * w + 3):
            continue
        keep[i] = True
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    if not mask.any():
        return mask

    # 膨胀后聚类，相近的笔画合为一组
    g = max(9, int(0.025 * max(h, w)))
    lab2, n2, st2 = _components(_dilate(mask, g))
    ink = np.bincount(lab2[mask > 0], minlength=n2)
    ink[0] = 0
    main = int(np.argmax(ink))
    group = {main}
    x0, y0 = st2[main, 0], st2[main, 1]
    x1, y1 = x0 + st2[main, 2], y0 + st2[main, 3]
    changed = True
    while changed:
        changed = False
        gh = y1 - y0
        px, py = int(0.8 * gh), int(0.3 * gh)
        for j in range(1, n2):
            if j in group or ink[j] < 0.03 * ink[main]:
                continue
            bx0, by0 = st2[j, 0], st2[j, 1]
            bx1, by1 = bx0 + st2[j, 2], by0 + st2[j, 3]
            if bx0 < x1 + px and bx1 > x0 - px and by0 < y1 + py and by1 > y0 - py:
                group.add(j)
                x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
                changed = True
    sel = np.zeros(n2, dtype=bool)
    sel[list(group)] = True
    return np.where(sel[lab2] & (mask > 0), 255, 0).astype(np.uint8)


FIT_MODES = {
    "紧贴签名": "tight",     # 等比缩放，裁掉两边空白，尺寸不超过 宽x高
    "固定画布": "canvas",    # 等比缩放，居中放在 宽x高 画布上
    "拉伸填满": "stretch",   # 拉伸到 宽x高，无空白但字形会变形
}


ROTATE_MODES = {
    "逆时针转正": "ccw",  # 竖向签名逆时针转 90°
    "顺时针转正": "cw",   # 竖向签名顺时针转 90°
    "不旋转": "none",
}
VERTICAL_RATIO = 1.3  # 签名高/宽超过该值视为竖向


def rotate_image(rgb, quarter_turns):
    """逆时针旋转 quarter_turns 个 90°。"""
    k = quarter_turns % 4
    return np.ascontiguousarray(np.rot90(rgb, k)) if k else rgb


def extract_signature(rgb, width=280, height=80, margin=2, sensitivity=50,
                      color="黑色", transparent=True, fit="tight", rotate="ccw"):
    """返回 (输出 RGBA 数组, 签名在原图中的外框 or None, 自动旋转角度 0/90/-90)。"""
    h, w = rgb.shape[:2]
    if max(h, w) > WORK_MAX_SIDE:
        s = WORK_MAX_SIDE / max(h, w)
        rgb = np.asarray(Image.fromarray(rgb).resize((int(w * s), int(h * s)), Image.BOX))
        h, w = rgb.shape[:2]

    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    norm = _normalize_background(gray)
    norm = np.asarray(Image.fromarray(norm).filter(
        ImageFilter.Kernel((3, 3), (1, 2, 1, 2, 4, 2, 1, 2, 1), 16)))

    t = float(np.clip(_otsu(norm), 90, 215)) + (sensitivity - 50) * 0.8
    t = float(np.clip(t, 30, 245))
    mask = np.where(norm < t, 255, 0).astype(np.uint8)
    mask = _select_signature(mask)

    canvas_rgb = (255, 255, 255)
    if not mask.any():
        out = np.zeros((height, width, 4), np.uint8)
        if not transparent:
            out[:] = (*canvas_rgb, 255)
        return out, None, 0

    # 平滑的透明度：墨迹越深越不透明，边缘抗锯齿
    ink_vals = norm[mask > 0].astype(np.float32)
    t_lo = float(np.percentile(ink_vals, 25))
    t_hi = min(250.0, t + 20.0)
    alpha = np.clip((t_hi - norm.astype(np.float32)) / max(t_hi - t_lo, 1.0), 0, 1)
    region = _dilate(mask, 3)
    alpha = np.where(region, alpha, 0) ** 0.8

    ys, xs = np.nonzero(alpha > 0.05)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    a = alpha[y0:y1, x0:x1]

    rgb_ink = INK_COLORS.get(color, (0, 0, 0))
    if rgb_ink is None:  # 原墨色：取笔画核心区域的平均颜色
        core = (mask > 0) & (norm <= np.percentile(ink_vals, 50))
        rgb_ink = tuple(int(v) for v in rgb[core].mean(axis=0))

    # 竖向签名（高明显大于宽）旋转 90° 转成水平
    turned = 0
    if rotate in ("ccw", "cw") and (y1 - y0) > VERTICAL_RATIO * (x1 - x0):
        turned = 90 if rotate == "ccw" else -90
        a = np.ascontiguousarray(np.rot90(a, 1 if rotate == "ccw" else 3))
    bh, bw = a.shape
    if fit == "stretch":  # 拉伸填满画布（会改变字形比例）
        nw, nh = width - 2 * margin, height - 2 * margin
    else:  # 等比缩放到不超过 宽x高
        scale = min((width - 2 * margin) / bw, (height - 2 * margin) / bh)
        nw, nh = max(1, round(bw * scale)), max(1, round(bh * scale))
        if fit == "tight":  # 画布紧贴签名，不留多余空白
            width, height = nw + 2 * margin, nh + 2 * margin
    nw, nh = max(1, nw), max(1, nh)
    resample = Image.BOX if nw * nh < bw * bh else Image.BICUBIC
    a = np.clip(np.asarray(Image.fromarray(a.astype(np.float32)).resize((nw, nh), resample)), 0, 1)

    full_a = np.zeros((height, width), np.float32)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    full_a[oy:oy + nh, ox:ox + nw] = a

    out = np.zeros((height, width, 4), np.uint8)
    if transparent:
        out[..., :3] = rgb_ink
        out[..., 3] = np.round(full_a * 255).astype(np.uint8)
    else:
        bg = np.array(canvas_rgb, np.float32)
        ink = np.array(rgb_ink, np.float32)
        out[..., :3] = np.round(bg * (1 - full_a[..., None]) + ink * full_a[..., None]).astype(np.uint8)
        out[..., 3] = 255
    return out, (x0, y0, x1, y1), turned


def output_path_for(src, out_dir, base_dir=None):
    stem = os.path.splitext(os.path.basename(src))[0]
    if out_dir:
        sub = ""
        if base_dir:
            rel = os.path.relpath(os.path.dirname(src), base_dir)
            sub = "" if rel == "." else rel
        d = os.path.join(out_dir, sub)
    else:
        d = os.path.join(os.path.dirname(src), OUT_DIR_NAME)
    return os.path.join(d, stem + OUT_SUFFIX + ".png")


def scan_folder(folder, recursive=True):
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d != OUT_DIR_NAME]
        for nm in sorted(names):
            if os.path.splitext(nm)[1].lower() in IMG_EXTS and OUT_SUFFIX + "." not in nm:
                files.append(os.path.join(root, nm))
        if not recursive:
            break
    return files


# ---------------------------------------------------------------- 界面

def checkerboard(w, h, cell=8):
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((xx // cell) + (yy // cell)) % 2 == 0, 255, 225).astype(np.uint8)
    return np.dstack([board] * 3)


class App(tk.Tk):
    def __init__(self, initial_paths=()):
        super().__init__()
        self.title("签名提取工具")
        # 打包后资源在 PyInstaller 的临时解压目录里
        icon = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
                            "assets", "icon.ico")
        if os.path.isfile(icon):
            self.iconbitmap(default=icon)
        # 高分屏下按系统缩放放大窗口，避免内容被挤出窗口
        k = max(1.0, self.winfo_fpixels("1i") / 96.0)
        ww = min(int(1000 * k), int(self.winfo_screenwidth() * 0.9))
        wh = min(int(720 * k), int(self.winfo_screenheight() * 0.85))
        self.geometry(f"{ww}x{wh}")
        self.minsize(int(820 * k), int(600 * k))
        self.files = []        # [(路径, 所属文件夹 or None)]
        self.busy = False
        self._preview_job = None
        self._tk_imgs = []

        self.var_w = tk.IntVar(value=280)
        self.var_h = tk.IntVar(value=80)
        self.var_margin = tk.IntVar(value=2)
        self.var_fit = tk.StringVar(value="紧贴签名")
        self.var_rotate = tk.StringVar(value="逆时针转正")
        self.rot = {}          # 路径 -> 手动旋转（逆时针 90° 的次数）
        self.var_sens = tk.IntVar(value=50)
        self.var_color = tk.StringVar(value="黑色")
        self.var_transparent = tk.BooleanVar(value=True)
        self.var_recursive = tk.BooleanVar(value=True)
        self.var_outdir = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="添加图片或文件夹开始")

        self._build()
        for v in (self.var_w, self.var_h, self.var_margin, self.var_sens,
                  self.var_color, self.var_transparent, self.var_fit, self.var_rotate):
            v.trace_add("write", lambda *_: self.schedule_preview())
        for p in initial_paths:
            if os.path.isdir(p):
                self._add_folder(p)
            elif os.path.isfile(p):
                self._add_files([p])

    def _build(self):
        pad = dict(padx=6, pady=4)
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="添加图片", command=self.on_add_files).pack(side="left")
        ttk.Button(top, text="添加文件夹", command=self.on_add_folder).pack(side="left", padx=4)
        ttk.Checkbutton(top, text="包含子文件夹", variable=self.var_recursive).pack(side="left", padx=4)
        ttk.Button(top, text="移除选中", command=self.on_remove).pack(side="left", padx=4)
        ttk.Button(top, text="清空列表", command=self.on_clear).pack(side="left")

        # 底部按钮栏先占位，保证窗口变小时不被挤掉
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", **pad)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, **pad)

        left = ttk.LabelFrame(body, text="图片列表")
        left.pack(side="left", fill="y")
        self.listbox = tk.Listbox(left, width=34, selectmode="extended", activestyle="none")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.schedule_preview())

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # 设置和结果从下往上排，原图预览占用剩余空间
        st = ttk.LabelFrame(right, text="设置")
        st.pack(side="bottom", fill="x", pady=(6, 0))

        rv = ttk.LabelFrame(right, text="提取结果（放大 2 倍显示，棋盘格为透明）")
        rv.pack(side="bottom", fill="x", pady=(6, 0))
        self.lbl_out = ttk.Label(rv, anchor="center")
        self.lbl_out.pack(fill="x", pady=6)

        pv = ttk.LabelFrame(right, text="原图（红框为识别到的签名区域）")
        pv.pack(fill="both", expand=True)
        rb = ttk.Frame(pv)
        rb.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Label(rb, text="手动旋转选中图片：").pack(side="left")
        for text, k in (("左转90°", 1), ("右转90°", 3), ("旋转180°", 2), ("复原", None)):
            ttk.Button(rb, text=text, command=lambda k=k: self.on_rotate(k)).pack(side="left", padx=2)
        ttk.Label(rb, text="（自动转正方向反了就点“旋转180°”）", foreground="#666").pack(side="left", padx=4)
        self.lbl_src = ttk.Label(pv, anchor="center")
        self.lbl_src.pack(fill="both", expand=True)
        r1 = ttk.Frame(st)
        r1.pack(fill="x", **pad)
        ttk.Label(r1, text="输出宽").pack(side="left")
        ttk.Spinbox(r1, from_=20, to=4000, width=6, textvariable=self.var_w).pack(side="left", padx=(2, 10))
        ttk.Label(r1, text="高").pack(side="left")
        ttk.Spinbox(r1, from_=10, to=4000, width=6, textvariable=self.var_h).pack(side="left", padx=(2, 10))
        ttk.Label(r1, text="边距").pack(side="left")
        ttk.Spinbox(r1, from_=0, to=100, width=4, textvariable=self.var_margin).pack(side="left", padx=(2, 10))
        ttk.Label(r1, text="墨色").pack(side="left")
        ttk.Combobox(r1, values=list(INK_COLORS), width=7, state="readonly",
                     textvariable=self.var_color).pack(side="left", padx=(2, 10))
        ttk.Radiobutton(r1, text="透明背景", value=True, variable=self.var_transparent).pack(side="left")
        ttk.Radiobutton(r1, text="白色背景", value=False, variable=self.var_transparent).pack(side="left", padx=4)

        r1b = ttk.Frame(st)
        r1b.pack(fill="x", **pad)
        ttk.Label(r1b, text="尺寸模式").pack(side="left")
        ttk.Combobox(r1b, values=list(FIT_MODES), width=9, state="readonly",
                     textvariable=self.var_fit).pack(side="left", padx=(2, 10))
        ttk.Label(r1b, text="竖向签名").pack(side="left")
        ttk.Combobox(r1b, values=list(ROTATE_MODES), width=10, state="readonly",
                     textvariable=self.var_rotate).pack(side="left", padx=(2, 10))
        ttk.Label(r1b, text="（签名高明显大于宽时自动转为水平）", foreground="#666").pack(side="left")

        r2 = ttk.Frame(st)
        r2.pack(fill="x", **pad)
        ttk.Label(r2, text="灵敏度").pack(side="left")
        ttk.Scale(r2, from_=0, to=100, variable=self.var_sens, length=220,
                  command=lambda v: self.var_sens.set(int(float(v)))).pack(side="left", padx=4)
        self.lbl_sens = ttk.Label(r2, width=4)
        self.lbl_sens.pack(side="left")
        ttk.Label(r2, text="（笔画断开/太淡调高；杂点多/背景脏调低）", foreground="#666").pack(side="left")

        r3 = ttk.Frame(st)
        r3.pack(fill="x", **pad)
        ttk.Label(r3, text="输出目录").pack(side="left")
        ttk.Entry(r3, textvariable=self.var_outdir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r3, text="浏览", command=self.on_pick_outdir).pack(side="left")
        ttk.Label(st, text=f"输出目录留空时，保存到原图所在文件夹下的“{OUT_DIR_NAME}”子文件夹",
                  foreground="#666").pack(anchor="w", padx=6, pady=(0, 4))

        ttk.Button(bottom, text="提取选中", command=lambda: self.on_run(True)).pack(side="left")
        ttk.Button(bottom, text="全部提取", command=lambda: self.on_run(False)).pack(side="left", padx=4)
        ttk.Button(bottom, text="打开输出目录", command=self.on_open_out).pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(bottom, textvariable=self.var_status, width=40).pack(side="left")

        self.var_sens.trace_add("write", lambda *_: self.lbl_sens.configure(text=str(self.var_sens.get())))
        self.lbl_sens.configure(text="50")

    # ---- 列表管理
    def _add_files(self, paths, base=None):
        known = {p for p, _ in self.files}
        for p in paths:
            p = os.path.normpath(p)
            if p in known or os.path.splitext(p)[1].lower() not in IMG_EXTS:
                continue
            self.files.append((p, base))
            known.add(p)
            self.listbox.insert("end", self._label(p, base))
        self.var_status.set(f"共 {len(self.files)} 张图片")
        if self.files and not self.listbox.curselection():
            self.listbox.selection_set(0)
            self.schedule_preview()

    def _add_folder(self, folder):
        folder = os.path.normpath(folder)
        found = scan_folder(folder, self.var_recursive.get())
        if not found:
            messagebox.showinfo("提示", "该文件夹下没有找到图片")
            return
        self._add_files(found, folder)

    def on_add_files(self):
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片", " ".join("*" + e for e in sorted(IMG_EXTS))), ("所有文件", "*.*")])
        if paths:
            self._add_files(paths)

    def on_add_folder(self):
        d = filedialog.askdirectory(title="选择文件夹")
        if d:
            self._add_folder(d)

    def _label(self, path, base):
        name = os.path.relpath(path, base) if base else os.path.basename(path)
        k = self.rot.get(path, 0) % 4
        return name + {0: "", 1: "  [↺90°]", 2: "  [180°]", 3: "  [↻90°]"}[k]

    def on_rotate(self, k):
        sel = self.listbox.curselection()
        if not sel:
            return
        for i in sel:
            path, base = self.files[i]
            self.rot[path] = 0 if k is None else (self.rot.get(path, 0) + k) % 4
            self.listbox.delete(i)
            self.listbox.insert(i, self._label(path, base))
        for i in sel:
            self.listbox.selection_set(i)
        self.schedule_preview()

    def on_remove(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            self.rot.pop(self.files[i][0], None)
            del self.files[i]
        self.var_status.set(f"共 {len(self.files)} 张图片")
        self.schedule_preview()

    def on_clear(self):
        self.listbox.delete(0, "end")
        self.files.clear()
        self.rot.clear()
        self.var_status.set("列表已清空")
        self.schedule_preview()

    def on_pick_outdir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.var_outdir.set(os.path.normpath(d))

    def on_open_out(self):
        d = self.var_outdir.get().strip()
        if not d:
            sel = self.listbox.curselection()
            if not self.files:
                return
            src = self.files[sel[0] if sel else 0][0]
            d = os.path.join(os.path.dirname(src), OUT_DIR_NAME)
        if os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showinfo("提示", "输出目录还不存在，请先提取")

    # ---- 参数
    def params(self):
        try:
            return dict(width=max(10, int(self.var_w.get())), height=max(10, int(self.var_h.get())),
                        margin=max(0, int(self.var_margin.get())), sensitivity=int(self.var_sens.get()),
                        color=self.var_color.get(), transparent=bool(self.var_transparent.get()),
                        fit=FIT_MODES.get(self.var_fit.get(), "tight"),
                        rotate=ROTATE_MODES.get(self.var_rotate.get(), "ccw"))
        except (tk.TclError, ValueError):
            return None

    # ---- 预览
    def schedule_preview(self):
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(200, self.update_preview)

    def update_preview(self):
        self._preview_job = None
        sel = self.listbox.curselection()
        prm = self.params()
        if not sel or prm is None or sel[0] >= len(self.files):
            self.lbl_src.configure(image="")
            self.lbl_out.configure(image="")
            return
        path = self.files[sel[0]][0]
        try:
            rgb = rotate_image(load_image(path), self.rot.get(path, 0))
            out, box, turned = extract_signature(rgb, **prm)
        except Exception as e:
            self.var_status.set(f"读取失败：{e}")
            return

        im = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        if box:
            s = 1.0 if max(h, w) <= WORK_MAX_SIDE else max(h, w) / WORK_MAX_SIDE
            x0, y0, x1, y1 = (int(v * s) for v in box)
            ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), outline=(230, 30, 30),
                                         width=max(2, int(max(h, w) / 300)))
        self.lbl_src.update_idletasks()
        aw = max(200, self.lbl_src.winfo_width() - 8)
        ah = max(150, self.lbl_src.winfo_height() - 8)
        im.thumbnail((aw, ah))

        ow, oh = out.shape[1], out.shape[0]
        zoom = 2 if ow * 2 <= 700 else max(0.1, 700 / ow)
        bg = Image.fromarray(checkerboard(ow, oh) if prm["transparent"] else
                             np.full((oh, ow, 3), 255, np.uint8)).convert("RGBA")
        res = Image.alpha_composite(bg, Image.fromarray(out, "RGBA")).convert("RGB")
        res = ImageOps.expand(res, border=1, fill=(160, 160, 160))
        res = res.resize((int(res.width * zoom), int(res.height * zoom)), Image.NEAREST)

        self._tk_imgs = [ImageTk.PhotoImage(im), ImageTk.PhotoImage(res)]
        self.lbl_src.configure(image=self._tk_imgs[0])
        self.lbl_out.configure(image=self._tk_imgs[1])
        if box is None:
            self.var_status.set("未检测到签名，可尝试调高灵敏度")
        elif turned:
            self.var_status.set("竖向签名，已自动%s旋转 90°" % ("逆时针" if turned > 0 else "顺时针"))
        else:
            self.var_status.set(os.path.basename(path))

    # ---- 批量处理
    def on_run(self, selected_only):
        if self.busy:
            return
        idx = self.listbox.curselection() if selected_only else range(len(self.files))
        items = [self.files[i] for i in idx]
        if not items:
            messagebox.showinfo("提示", "请先添加或选中图片")
            return
        prm = self.params()
        if prm is None:
            messagebox.showerror("错误", "设置中的数字无效")
            return
        out_dir = self.var_outdir.get().strip()
        self.busy = True
        self.progress.configure(maximum=len(items), value=0)
        threading.Thread(target=self._worker, args=(items, prm, out_dir, dict(self.rot)),
                         daemon=True).start()

    def _worker(self, items, prm, out_dir, rots):
        ok, empty, failed = 0, [], []
        for n, (path, base) in enumerate(items, 1):
            try:
                rgb = rotate_image(load_image(path), rots.get(path, 0))
                out, box, _ = extract_signature(rgb, **prm)
                if box is None:
                    empty.append(path)
                else:
                    dst = output_path_for(path, out_dir, base)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    im = Image.fromarray(out, "RGBA")
                    (im if prm["transparent"] else im.convert("RGB")).save(dst)
                    ok += 1
            except Exception as e:
                failed.append(f"{os.path.basename(path)}：{e}")
            self.after(0, self._progress, n, len(items), os.path.basename(path))
        self.after(0, self._done, ok, empty, failed)

    def _progress(self, n, total, name):
        self.progress.configure(value=n)
        self.var_status.set(f"{n}/{total}  {name}")

    def _done(self, ok, empty, failed):
        self.busy = False
        msg = f"完成：成功 {ok} 张"
        if empty:
            msg += f"\n未检测到签名 {len(empty)} 张：\n" + "\n".join(os.path.basename(p) for p in empty[:10])
        if failed:
            msg += f"\n失败 {len(failed)} 张：\n" + "\n".join(failed[:10])
        self.var_status.set(msg.splitlines()[0])
        messagebox.showinfo("完成", msg)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # 命令行：signature_extractor --cli 输入图片或文件夹... [-o 输出目录]
        args = sys.argv[2:]
        out_dir = ""
        if "-o" in args:
            i = args.index("-o")
            out_dir = args[i + 1]
            args = args[:i] + args[i + 2:]
        for a in args:
            items = [(p, a) for p in scan_folder(a)] if os.path.isdir(a) else [(a, None)]
            for p, base in items:
                out, box, _ = extract_signature(load_image(p))
                if box is None:
                    print("未检测到签名:", p)
                    continue
                dst = output_path_for(p, out_dir, base)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                Image.fromarray(out, "RGBA").save(dst)
                print("已保存:", dst)
        return
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App(sys.argv[1:]).mainloop()


if __name__ == "__main__":
    main()
