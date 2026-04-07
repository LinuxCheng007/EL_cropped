import sys, os, json, threading

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET = os.path.join(_ROOT, 'runtime', 'Lib', 'site-packages')
if os.path.isdir(_TARGET) and _TARGET not in sys.path:
    sys.path.insert(0, _TARGET)
_PYDIR = os.path.join(_ROOT, 'runtime')
for _sp in ['Lib\\site-packages', 'site-packages']:
    _p = os.path.join(_PYDIR, _sp)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
except ImportError:
    print('[ERROR] numpy not found. Run 2_install_libraries.bat')
    input('Press Enter...'); sys.exit(1)
try:
    import cv2
except ImportError:
    print('[ERROR] opencv not found. Run 2_install_libraries.bat')
    input('Press Enter...'); sys.exit(1)
try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print('[ERROR] flask not found. Run 2_install_libraries.bat')
    input('Press Enter...'); sys.exit(1)

import base64, time, traceback, webbrowser
import concurrent.futures

BASE    = os.path.dirname(os.path.abspath(__file__))
HTML    = os.path.join(BASE, 'index.html')
CORR_F  = os.path.join(BASE, 'corrections.json')
_lock   = threading.Lock()

USE_GPU = cv2.ocl.haveOpenCL()
if USE_GPU:
    cv2.ocl.setUseOpenCL(True)
WORKERS = min(os.cpu_count() or 4, 8)
pool    = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
app     = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# ================================================================
# 检测内核 v4.0
#
# 相比 v3.12 的改进：
#   [H] 新增 Hough 直线检测法作为主检测方法
#       → 直接找面板边框的四条直线，交点即角点
#       → 底边选择：从亮区中心向下，取第一条跨越面板35%宽度的长线
#         确保定位到面板边框而非下方平台
#       → 左右边：取最外侧的长竖线（面板边框而非内部栅线）
#   [I] 保留亮度法+纹理法+边缘法作为 Hough 失败时的备用
#   [J] 默认 pad 改为 0.03（3%黑边），原值 0.065~0.07
#   [K] 亮度法备用时也增加底边约束，防止延伸到平台
#   
# ================================================================

def order_corners(pts):
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

def _hull_to_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)

def _clean(gray, top_frac=0.03):
    sh, sw = gray.shape
    g = gray.copy()
    g[:int(sh * top_frac), :] = 0
    if np.any(g > 20):
        p98 = float(np.percentile(g[g > 0], 98.5))
        hot = (g > max(p98 * 0.92, 20)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(hot)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < sw * sh * 0.008:
                g[labels == i] = 0
    return g


# ================================================================
# [H] Hough 直线检测法（主检测方法）
# ================================================================

def _method_hough(gray, sh, sw, bcx, bcy):
    """
    Hough 直线检测：找面板边框的四条边。

    关键策略：
    - 底/顶边：从亮区中心向外扫描，取第一条"长线"（>35%面板宽度）
      → 跳过内部电池片栅线（短），精确定位到面板边框
      → 不会延伸到平台（平台边缘在面板边框之后）
    - 左/右边：取最外侧的长竖线（>15%面板高度）
    """
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 20, 60)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    min_ll = int(min(sw, sh) * 0.10)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_ll, maxLineGap=25)
    if lines is None or len(lines) < 4:
        return None

    lines_arr = lines.reshape(-1, 4)
    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lines_arr:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.hypot(x2 - x1, y2 - y1)
        if angle < 20 or angle > 160:
            h_lines.append(((y1 + y2) / 2, length, x1, y1, x2, y2))
        elif 70 < angle < 110:
            v_lines.append(((x1 + x2) / 2, length, x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    long_h = sw * 0.35   # 面板边框 = 跨越35%宽度以上的水平线
    long_v = sh * 0.15   # 面板边框 = 跨越15%高度以上的垂直线

    # ── 底边：从中心向下，第一条长水平线 ──
    bot_sorted = sorted(
        [t for t in h_lines if t[0] > bcy + sh * 0.05],
        key=lambda t: t[0])
    bot_line = None
    for t in bot_sorted:
        if t[1] >= long_h:
            bot_line = t; break
    if bot_line is None:
        # 无长线，取最近的中等线
        bot_med = [t for t in bot_sorted if t[1] >= min_ll]
        bot_line = bot_med[0] if bot_med else None
    if bot_line is None:
        return None

    # ── 顶边：从中心向上，第一条长水平线 ──
    top_sorted = sorted(
        [t for t in h_lines if t[0] < bcy - sh * 0.05],
        key=lambda t: -t[0])
    top_line = None
    for t in top_sorted:
        if t[1] >= long_h:
            top_line = t; break
    if top_line is None:
        top_med = [t for t in top_sorted if t[1] >= min_ll]
        top_line = top_med[0] if top_med else None
    if top_line is None:
        return None

    # ── 左边：最外侧（最小 x）的长竖线 ──
    left_sorted = sorted(
        [t for t in v_lines if t[0] < bcx - sw * 0.05],
        key=lambda t: t[0])
    left_line = None
    for t in left_sorted:
        if t[1] >= long_v:
            left_line = t; break
    if left_line is None:
        return None

    # ── 右边：最外侧（最大 x）的长竖线 ──
    right_sorted = sorted(
        [t for t in v_lines if t[0] > bcx + sw * 0.05],
        key=lambda t: -t[0])
    right_line = None
    for t in right_sorted:
        if t[1] >= long_v:
            right_line = t; break
    if right_line is None:
        return None

    # 四线求交
    def seg_to_line(x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        ln = max(np.hypot(vx, vy), 1e-6)
        return [vx / ln, vy / ln, (x1 + x2) / 2, (y1 + y2) / 2]

    def line_intersect(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    tl = seg_to_line(*top_line[2:])
    bl = seg_to_line(*bot_line[2:])
    ll = seg_to_line(*left_line[2:])
    rl = seg_to_line(*right_line[2:])

    corners = [
        line_intersect(tl, ll),   # TL
        line_intersect(tl, rl),   # TR
        line_intersect(bl, rl),   # BR
        line_intersect(bl, ll),   # BL
    ]
    if any(c is None for c in corners):
        return None

    pts = order_corners(np.array(corners, dtype=np.float32))

    # 宽高比验证
    w1 = np.linalg.norm(pts[1] - pts[0])
    h1 = np.linalg.norm(pts[3] - pts[0])
    ar = w1 / max(h1, 1e-6)
    if ar < 1.4 or ar > 3.5:
        return None

    return pts


# ================================================================
# 备用检测方法（原 v3.12 保留）
# ================================================================

def _method_brightness(gray, sh, sw, gray_blur=None):
    """亮度 Otsu 阈值法"""
    blur = gray_blur if gray_blur is not None else cv2.GaussianBlur(_clean(gray), (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    best_pts, best_score = None, -1
    for frac in [1.0, 0.90, 0.82, 0.74, 0.66, 0.58, 0.50, 0.42, 0.35]:
        thresh = max(8, otsu_val * frac)
        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.zeros_like(mask)
        m[labels == largest] = 255
        ys, xs = np.where(m > 0)
        if len(xs) < 40:
            continue
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
        area = cv2.contourArea(hull)
        if area < sw * sh * 0.08 or area > sw * sh * 0.92:
            continue
        pts = _hull_to_quad(hull)
        o = order_corners(pts)
        w1 = np.linalg.norm(o[1] - o[0])
        w2 = np.linalg.norm(o[2] - o[3])
        h1 = np.linalg.norm(o[3] - o[0])
        h2 = np.linalg.norm(o[2] - o[1])
        ar = max(w1, w2) / max(max(h1, h2), 1e-6)
        if ar < 1.4 or ar > 3.5:
            continue
        rect = (min(w1, w2) / max(w1, w2)) * (min(h1, h2) / max(h1, h2))
        score = area * rect * max(1.0 - abs(ar - 2.1) / 2.0, 0.3)
        if score > best_score:
            best_score, best_pts = score, pts

    return order_corners(best_pts) if best_pts is not None else None


def _refine_corners_by_lines(gray, rough_pts, sw, sh):
    """Canny + fitLine 精确化角点"""
    blur = cv2.GaussianBlur(_clean(gray), (5, 5), 0)
    edges = cv2.Canny(blur, 25, 75)
    margin_n = max(14, int(min(sw, sh) * 0.028))
    quad_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(quad_mask, [rough_pts.astype(np.int32)], 255)
    expand_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (margin_n * 2 + 1, margin_n * 2 + 1))
    quad_mask = cv2.dilate(quad_mask, expand_k, iterations=1)
    edges = cv2.bitwise_and(edges, quad_mask)
    ys, xs = np.where(edges > 0)
    if len(xs) < 40:
        return rough_pts

    pts_arr = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    o = order_corners(rough_pts)
    fitted = []
    for i in range(4):
        p1, p2 = o[i], o[(i + 1) % 4]
        dx, dy = p2 - p1
        length = max(np.hypot(dx, dy), 1e-6)
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        rel = pts_arr - p1
        d_n = rel[:, 0] * nx + rel[:, 1] * ny
        d_t = rel[:, 0] * tx + rel[:, 1] * ty
        margin_t = length * 0.10
        sel = (np.abs(d_n) < margin_n) & \
              (d_t > -margin_t) & (d_t < length + margin_t)
        side_pts = pts_arr[sel]
        if len(side_pts) < 6:
            fitted.append(None)
            continue
        line = cv2.fitLine(side_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        fitted.append(line)

    def _line_intersect(l1, l2):
        if l1 is None or l2 is None:
            return None
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    refined = []
    max_shift = min(sw, sh) * 0.07
    for i in range(4):
        pt = _line_intersect(fitted[(i - 1) % 4], fitted[i])
        if pt is not None:
            dist = np.hypot(pt[0] - o[i][0], pt[1] - o[i][1])
            if dist < max_shift:
                refined.append(pt)
                continue
        refined.append(o[i])
    return np.array(refined, dtype=np.float32)


# ================================================================
# 主检测入口
# ================================================================

def detect_corners_v2(img, pad_pct=0.03):
    """
    v4.0 检测流程：
      1. Hough 直线法（主）→ 直接找面板边框四条线
      2. 亮度法（备）→ Hough 失败时启用
      3. fitLine 精确化角点
      4. pad 外扩 + clip
    """
    H, W = img.shape[:2]
    sc = min(1.0, 1200 / max(W, H))
    sm = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)

    # 预计算亮区中心
    cleaned = _clean(gray)
    gray_blur = cv2.GaussianBlur(cleaned, (31, 31), 0)
    otsu_val, _ = cv2.threshold(gray_blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bright_mask = cv2.threshold(gray_blur, otsu_val * 0.6, 255,
                                  cv2.THRESH_BINARY)
    ys, xs = np.where(bright_mask > 0)
    if len(xs) < 100:
        return None
    bcx, bcy = float(np.mean(xs)), float(np.mean(ys))

    # ── 方法 1：Hough 直线检测（主）──
    best_pts = _method_hough(gray, sh, sw, bcx, bcy)
    method = 'hough'

    # ── 方法 2：亮度阈值法（备）──
    if best_pts is None:
        best_pts = _method_brightness(gray, sh, sw, gray_blur)
        method = 'brightness'

    if best_pts is None:
        return None

    # fitLine 精确化
    best_pts = _refine_corners_by_lines(gray, order_corners(best_pts), sw, sh)
    best_pts = order_corners(best_pts)

    # [J] pad 外扩（四边均匀 3%）
    cx, cy = best_pts.mean(axis=0)
    h_side = max(np.linalg.norm(best_pts[3] - best_pts[0]),
                 np.linalg.norm(best_pts[2] - best_pts[1]))
    expand_px = round(h_side * pad_pct)
    for i in range(4):
        dx, dy = best_pts[i] - np.array([cx, cy])
        ln = max(np.hypot(dx, dy), 1e-6)
        best_pts[i] = best_pts[i] + np.array([dx, dy]) / ln * expand_px

    for i in range(4):
        best_pts[i, 0] = np.clip(best_pts[i, 0], 0, sw - 1)
        best_pts[i, 1] = np.clip(best_pts[i, 1], 0, sh - 1)

    # 缩放回原始分辨率
    inv = 1.0 / sc
    pts = order_corners(best_pts * inv)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


def warp_image(img, pts):
    tl, tr, br, bl = pts
    dw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    dh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if dw < 10 or dh < 10:
        raise ValueError('Invalid corners')
    dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    if USE_GPU:
        return cv2.warpPerspective(cv2.UMat(img), M, (dw, dh)).get()
    return cv2.warpPerspective(img, M, (dw, dh))


def process_image(img_bytes, pad_pct, forced=None, img_name=None):
    """
    处理优先级：
      1. forced（前端手动编辑角点） → 直接使用
      2. corrections.json 中有该文件名的 manual 记录 → 缩放后使用
      3. 自动检测（Hough 主 + 亮度备） + 学习偏差
    """
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, 'Decode failed'
        H, W = img.shape[:2]

        if forced is not None:
            pts = np.array(forced, dtype=np.float32)

        elif img_name:
            with _lock:
                corr = _load_corrections()
            corr_map = {c['name']: c for c in corr}
            if img_name in corr_map:
                rec = corr_map[img_name]
                orig_W, orig_H = rec.get('img_w', 0), rec.get('img_h', 0)
                if orig_W == 0 or orig_H == 0:
                    chk = np.array(rec['manual'], dtype=np.float32)
                    if chk[:, 0].max() > 4000 or chk[:, 1].max() > 3000:
                        orig_W, orig_H = 6000, 4000
                if orig_W > 0 and orig_H > 0:
                    pts = np.array(rec['manual'], dtype=np.float32)
                    pts[:, 0] *= W / orig_W
                    pts[:, 1] *= H / orig_H
                    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                    pts = order_corners(pts)
                else:
                    pts = detect_corners_v2(img, pad_pct)
                    if pts is None:
                        return None, None, 'Panel not detected'
                    pts = _apply_learned_bias(pts, W, H)
            else:
                pts = detect_corners_v2(img, pad_pct)
                if pts is None:
                    return None, None, 'Panel not detected'
                pts = _apply_learned_bias(pts, W, H)
        else:
            pts = detect_corners_v2(img, pad_pct)
            if pts is None:
                return None, None, 'Panel not detected'
            pts = _apply_learned_bias(pts, W, H)

        warped = warp_image(img, pts)
        ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (buf.tobytes(), pts.tolist(), None) if ok else (None, None, 'Encode failed')
    except Exception as e:
        return None, None, str(e)


# ================================================================
# 学习系统 — 偏差计算与应用
# ================================================================

_learned_bias = {'top': 0.0, 'bot': 0.0, 'left': 0.0, 'right': 0.0, 'n': 0}
_bias_lock = threading.Lock()

def _update_learned_bias():
    """从 corrections.json 重新计算四边学习偏差，至少 3 条记录才生效"""
    global _learned_bias
    with _lock:
        corr = _load_corrections()
    if not corr:
        with _bias_lock:
            _learned_bias = {'top': 0.0, 'bot': 0.0, 'left': 0.0, 'right': 0.0, 'n': 0}
        return
    top_l, bot_l, left_l, right_l = [], [], [], []
    for c in corr:
        auto   = np.array(c['auto'],   dtype=np.float32)
        manual = np.array(c['manual'], dtype=np.float32)
        W, H   = c['img_w'], c['img_h']
        if W == 0 or H == 0:
            continue
        delta = manual - auto
        top_l.append(  ((delta[0][1] + delta[1][1]) / 2) / H )
        bot_l.append(  ((delta[2][1] + delta[3][1]) / 2) / H )
        left_l.append( ((delta[0][0] + delta[3][0]) / 2) / W )
        right_l.append(((delta[1][0] + delta[2][0]) / 2) / W )
    with _bias_lock:
        _learned_bias = {
            'top':   float(np.mean(top_l)),
            'bot':   float(np.mean(bot_l)),
            'left':  float(np.mean(left_l)),
            'right': float(np.mean(right_l)),
            'n':     len(corr)
        }
    b = _learned_bias
    print('  [学习] 偏差已更新 (n=%d): 上=%.4f 下=%.4f 左=%.4f 右=%.4f' % (
          b['n'], b['top'], b['bot'], b['left'], b['right']))

def _apply_learned_bias(pts, W, H):
    """将学习偏差叠加到检测角点 (TL TR BR BL)，≥3 条记录才生效"""
    with _bias_lock:
        bias = dict(_learned_bias)
    if bias['n'] < 3:
        return pts
    pts = pts.copy()
    pts[0][1] += bias['top']   * H
    pts[1][1] += bias['top']   * H
    pts[2][1] += bias['bot']   * H
    pts[3][1] += bias['bot']   * H
    pts[0][0] += bias['left']  * W
    pts[3][0] += bias['left']  * W
    pts[1][0] += bias['right'] * W
    pts[2][0] += bias['right'] * W
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


# ================================================================
# 标注学习系统
# ================================================================

def _load_corrections():
    if os.path.exists(CORR_F):
        try:
            with open(CORR_F, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_corrections(data):
    with open(CORR_F, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _compute_analysis(corrections):
    """分析所有手动修正，计算四边偏差、建议 pad、最差图片"""
    if not corrections:
        return None
    top_biases, bot_biases, left_biases, right_biases = [], [], [], []
    outward_projs, pad_corrections = [], []
    worst = []
    for c in corrections:
        auto = np.array(c['auto'], dtype=np.float32)
        manual = np.array(c['manual'], dtype=np.float32)
        W, H = c['img_w'], c['img_h']
        if W == 0 or H == 0:
            continue
        delta = manual - auto
        top_biases.append((delta[0][1] + delta[1][1]) / 2)
        bot_biases.append((delta[2][1] + delta[3][1]) / 2)
        left_biases.append((delta[0][0] + delta[3][0]) / 2)
        right_biases.append((delta[1][0] + delta[2][0]) / 2)
        centroid = auto.mean(axis=0)
        dirs = auto - centroid
        norms = np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
        dirs = dirs / norms
        outward = (delta * dirs).sum(axis=1).mean()
        outward_projs.append(outward)
        pad_corrections.append(outward / H)
        mean_err = float(np.abs(delta).mean())
        worst.append({'name': c.get('name', '?'), 'err': round(mean_err, 1)})

    def m(lst):
        clean = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else 0.0

    current_pad   = corrections[-1].get('pad', 0.03)
    mean_pad_corr = m(pad_corrections)
    suggested_pad = round(max(0.0, min(0.15, current_pad + mean_pad_corr)), 3)
    worst.sort(key=lambda x: x['err'], reverse=True)
    return {
        'count': len(corrections),
        'edge': {
            'top':   round(m(top_biases), 1),
            'bot':   round(m(bot_biases), 1),
            'left':  round(m(left_biases), 1),
            'right': round(m(right_biases), 1),
        },
        'mean_outward':  round(m(outward_projs), 1),
        'current_pad':   round(current_pad, 4),
        'suggested_pad': suggested_pad,
        'pad_delta':     round(mean_pad_corr, 4),
        'worst':         worst[:5],
    }


# ================================================================
# Flask routes
# ================================================================

@app.route('/')
def index():
    with open(HTML, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html; charset=utf-8')

@app.route('/info')
def info():
    dev = ''
    if USE_GPU:
        try: dev = cv2.ocl.Device.getDefault().name()
        except: dev = 'OpenCL'
    with _lock:
        corr = _load_corrections()
    return jsonify({'gpu': USE_GPU, 'workers': WORKERS,
                    'cv': cv2.__version__, 'device': dev,
                    'corrections': len(corr)})

@app.route('/process', methods=['POST'])
def process():
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f: return jsonify({'ok': False, 'error': 'No image'})
        pad      = float(request.form.get('pad', 0.03))
        forced   = json.loads(request.form['corners']) \
                   if request.form.get('corners') else None
        img_name = request.form.get('name', None)
        jpeg, corners, err = process_image(f.read(), pad, forced, img_name)
        if err: return jsonify({'ok': False, 'error': err})
        return jsonify({'ok': True,
                        'jpeg_b64': base64.b64encode(jpeg).decode(),
                        'corners': corners,
                        'ms': round((time.time()-t0)*1000)})
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})

@app.route('/batch', methods=['POST'])
def batch():
    t0  = time.time()
    fs  = request.files.getlist('images')
    pad = float(request.form.get('pad', 0.03))
    forced = json.loads(request.form['corners']) \
             if request.form.get('corners') else None
    names_raw = request.form.get('names', None)
    names = json.loads(names_raw) if names_raw else [None] * len(fs)
    if len(names) != len(fs):
        names = [None] * len(fs)

    def proc(f, name):
        return process_image(f.read(), pad, forced, name)

    futures = {pool.submit(proc, f, n): i for i, (f, n) in enumerate(zip(fs, names))}
    results = [None] * len(fs)
    for fut, idx in futures.items():
        jpeg, corners, err = fut.result()
        results[idx] = {'ok': False, 'error': err} if err else \
                       {'ok': True,
                        'jpeg_b64': base64.b64encode(jpeg).decode(),
                        'corners': corners}
    return jsonify({'results': results, 'ms': round((time.time()-t0)*1000)})

@app.route('/correction', methods=['POST'])
def save_correction():
    try:
        d = request.get_json(force=True)
        auto   = np.array(d['auto'],   dtype=np.float32)
        manual = np.array(d['manual'], dtype=np.float32)
        record = {
            'name':   d.get('name', 'unknown'),
            'img_w':  int(d.get('img_w', 0)),
            'img_h':  int(d.get('img_h', 0)),
            'pad':    float(d.get('pad', 0.03)),
            'auto':   auto.tolist(),
            'manual': manual.tolist(),
            'ts':     int(time.time()),
        }
        with _lock:
            corr = _load_corrections()
            corr = [c for c in corr if c.get('name') != record['name']]
            corr.append(record)
            _save_corrections(corr)
        _update_learned_bias()
        return jsonify({'ok': True, 'saved': True, 'total': len(corr)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/analysis')
def analysis():
    try:
        with _lock:
            corr = _load_corrections()
        result = _compute_analysis(corr)
        if result is None:
            return jsonify({'ok': True, 'count': 0, 'ready': False})
        result['ok'] = True
        result['ready'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/corrections/clear', methods=['POST'])
def clear_corrections():
    with _lock:
        _save_corrections([])
    return jsonify({'ok': True})


# ================================================================
# 会话持久化 — 支持关闭网页后恢复上次进度
# ================================================================

import shutil, glob, re as _re

SESS_DIR = os.path.join(BASE, 'sessions')
os.makedirs(SESS_DIR, exist_ok=True)
_sid_re = _re.compile(r'^[0-9a-fA-F\-]{8,64}$')

def _sess_path(sid):
    if not _sid_re.match(sid):
        return None
    return os.path.join(SESS_DIR, sid)

@app.route('/session/save', methods=['POST'])
def session_save():
    try:
        d = request.get_json(force=True)
        sid = d.get('sid', '')
        sp = _sess_path(sid)
        if not sp:
            return jsonify({'ok': False, 'error': 'invalid sid'})
        os.makedirs(sp, exist_ok=True)
        meta_file = os.path.join(sp, 'meta.json')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/session/save_img', methods=['POST'])
def session_save_img():
    try:
        sid = request.form.get('sid', '')
        idx = request.form.get('idx', '')
        sp = _sess_path(sid)
        if not sp:
            return jsonify({'ok': False, 'error': 'invalid sid'})
        os.makedirs(sp, exist_ok=True)
        img = request.files.get('image')
        if not img:
            return jsonify({'ok': False, 'error': 'no image'})
        img.save(os.path.join(sp, 'img_%s.jpg' % idx))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/session/load/<sid>')
def session_load(sid):
    sp = _sess_path(sid)
    if not sp:
        return jsonify({'ok': True, 'exists': False})
    meta_file = os.path.join(sp, 'meta.json') if sp else ''
    if not os.path.exists(meta_file):
        return jsonify({'ok': True, 'exists': False})
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        existing = []
        for fn in glob.glob(os.path.join(sp, 'img_*.jpg')):
            base = os.path.basename(fn)
            try:
                idx = int(base.replace('img_', '').replace('.jpg', ''))
                existing.append(idx)
            except ValueError:
                pass
        meta['existing_imgs'] = existing
        return jsonify({'ok': True, 'exists': True, 'meta': meta})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/session/img/<sid>/<int:idx>')
def session_img(sid, idx):
    sp = _sess_path(sid)
    if not sp:
        return ('', 404)
    img_path = os.path.join(sp, 'img_%d.jpg' % idx)
    if not os.path.exists(img_path):
        return ('', 404)
    with open(img_path, 'rb') as f:
        data = f.read()
    return Response(data, mimetype='image/jpeg')

@app.route('/session/clear/<sid>', methods=['POST'])
def session_clear(sid):
    sp = _sess_path(sid)
    if sp and os.path.isdir(sp):
        shutil.rmtree(sp, ignore_errors=True)
    return jsonify({'ok': True})


def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:15789')

if __name__ == '__main__':
    import threading as _th
    _th.Thread(target=open_browser, daemon=True).start()
    gpu_str = ('GPU: ' + (cv2.ocl.Device.getDefault().name()
               if USE_GPU else '')) if USE_GPU else 'CPU mode'
    n_corr = len(_load_corrections())
    _update_learned_bias()
    print('=' * 56)
    print('  华矩EL 裁剪工具  v5-UI + 检测内核 v4.0 + 学习系统')
    print('  ' + gpu_str)
    print('  线程数: %d   OpenCV: %s' % (WORKERS, cv2.__version__))
    print('  检测方法: Hough直线(主) + 亮度阈值(备)')
    print('  默认外扩: 3%% 黑边')
    print('  已积累标注: %d 张  (≥3张时学习偏差自动生效)' % n_corr)
    print('  局域网: http://192.168.3.119:15789')
    print('  关闭此窗口退出')
    print('=' * 56)
    app.run(host='0.0.0.0', port=15789, debug=False,
            use_reloader=False, threaded=True)
