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
    from flask import Flask, request, jsonify, Response, session
except ImportError:
    print('[ERROR] flask not found. Run 2_install_libraries.bat')
    input('Press Enter...'); sys.exit(1)

import base64, time, traceback, hashlib
import concurrent.futures

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
HTML    = os.path.join(BASE, 'index.html')
CORR_F  = os.path.join(BASE, 'corrections.json')
CORR_DRONE_F = os.path.join(BASE, 'corrections_drone.json')
USERS_F = os.path.join(ROOT, 'users.json')
_lock   = threading.Lock()

USE_GPU = cv2.ocl.haveOpenCL()
if USE_GPU:
    cv2.ocl.setUseOpenCL(True)
WORKERS = min(os.cpu_count() or 4, 8)
pool    = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
app     = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get('EL_SECRET_KEY', 'el-crop-tool-local-session-v1')

# ================================================================
# 用户认证管理（基于 users.json）
# ================================================================

def _new_salt():
    return base64.b64encode(os.urandom(16)).decode()

def _hash_password(password, salt):
    combined = salt + password
    return base64.b64encode(hashlib.sha256(combined.encode()).digest()).decode()

def _load_users():
    try:
        if os.path.exists(USERS_F):
            with open(USERS_F, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                return data.get('users', [])
    except: pass
    return []

def _save_users(users):
    try:
        with open(USERS_F, 'w', encoding='utf-8') as f:
            json.dump({'users': users}, f, ensure_ascii=False, indent=2)
    except: pass

def _init_default_admin():
    users = _load_users()
    if not users:
        salt = _new_salt()
        pw_hash = _hash_password('hjjc', salt)
        users.append({
            'username': 'admin',
            'password_hash': pw_hash,
            'salt': salt,
            'role': 'admin',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        _save_users(users)

def _validate_user(username, password):
    users = _load_users()
    for u in users:
        if u['username'] == username:
            return _hash_password(password, u['salt']) == u['password_hash']
    return False

def _user_role(username):
    users = _load_users()
    for u in users:
        if u['username'] == username:
            return u.get('role', 'user')
    return None

def _add_user(username, password, role='user'):
    users = _load_users()
    for u in users:
        if u['username'] == username:
            return False, '用户名已存在'
    if len(username) < 2:
        return False, '用户名至少2个字符'
    if len(password) < 4:
        return False, '密码至少4个字符'
    salt = _new_salt()
    pw_hash = _hash_password(password, salt)
    users.append({
        'username': username,
        'password_hash': pw_hash,
        'salt': salt,
        'role': role,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    _save_users(users)
    return True, '添加成功'

def _remove_user(username):
    users = _load_users()
    users = [u for u in users if u['username'] != username]
    _save_users(users)

def _change_password(username, new_password):
    users = _load_users()
    for u in users:
        if u['username'] == username:
            u['salt'] = _new_salt()
            u['password_hash'] = _hash_password(new_password, u['salt'])
            _save_users(users)
            return True
    return False

# Initialize default admin on first run
_init_default_admin()

AUTH_FREE_PATHS = {'/', '/logo', '/health', '/auth/status', '/auth/login', '/auth/logout',
                   '/auth/recover', '/auth/check-username'}


@app.before_request
def require_login():
    if request.path in AUTH_FREE_PATHS:
        return None
    if session.get('logged_in') is True:
        return None
    return jsonify({'ok': False, 'auth': False, 'error': 'login required'}), 401


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

def _expand_corners(pts, h_pad, v_pad=None):
    """
    外扩角点：h_pad 控制左右方向外扩比例，v_pad 控制上下方向外扩比例。
    若 v_pad 为 None 则与 h_pad 相同（保持向后兼容）。
    """
    if v_pad is None:
        v_pad = h_pad
    cx, cy = pts.mean(axis=0)
    h_side = max(np.linalg.norm(pts[3] - pts[0]),
                  np.linalg.norm(pts[2] - pts[1]))
    expand_x = round(h_side * h_pad)
    expand_y = round(h_side * v_pad)
    result = pts.copy()
    for i in range(4):
        dx, dy = result[i] - np.array([cx, cy])
        ln = max(np.hypot(dx, dy), 1e-6)
        result[i] = result[i] + np.array([dx / ln * expand_x, dy / ln * expand_y])
    return result

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

def detect_corners_v2(img, pad_pct=0.005, v_pad=None):
    """
    v4.0 检测流程：
      1. Hough 直线法（主）→ 直接找面板边框四条线
      2. 亮度法（备）→ Hough 失败时启用
      3. fitLine 精确化角点
      4. pad 外扩 + clip（左右 h_pad，上下 v_pad）
    """
    if v_pad is None:
        v_pad = pad_pct
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

    # pad 外扩（默认 pad_pct=0.005 → 0.5%）
    best_pts = _expand_corners(best_pts, pad_pct, v_pad)

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


def process_image(img_bytes, pad_pct, forced=None, img_name=None, v_pad=None):
    """
    处理优先级：
      1. forced（前端手动编辑角点） → 直接使用
      2. corrections.json 中有该文件名的 manual 记录 → 缩放后使用
      3. 自动检测（Hough 主 + 亮度备） + 学习偏差
    """
    if v_pad is None:
        v_pad = pad_pct
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
                    pts = detect_corners_v2(img, pad_pct, v_pad)
                    if pts is None:
                        return None, None, 'Panel not detected'
                    pts = _apply_learned_bias(pts, W, H)
            else:
                pts = detect_corners_v2(img, pad_pct, v_pad)
                if pts is None:
                    return None, None, 'Panel not detected'
                pts = _apply_learned_bias(pts, W, H)
        else:
            pts = detect_corners_v2(img, pad_pct, v_pad)
            if pts is None:
                return None, None, 'Panel not detected'
            pts = _apply_learned_bias(pts, W, H)

        warped = warp_image(img, pts)
        ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (buf.tobytes(), pts.tolist(), None) if ok else (None, None, 'Encode failed')
    except Exception as e:
        return None, None, str(e)


# ================================================================
# 无人机拍摄模式 — 独立检测内核
# 一张无人机照片包含3个并排面板，裁出左/中/右三张
# ================================================================

def _drone_find_dominant_row(gray, x_left, x_right, y_top, y_bot):
    """
    用行亮度投影找"最主导的面板行"的垂直范围。
    适用于图像里多行面板堆叠的场景 —— 只保留最大/最亮的那一行。
    返回 (y_top_panel, y_bot_panel)。
    """
    col_strip = gray[y_top:y_bot, x_left:x_right]
    if col_strip.shape[0] < 50:
        return y_top, y_bot
    row_mean = col_strip.astype(np.float32).mean(axis=1)
    ksize = max(31, int(len(row_mean) * 0.02)) | 1
    row_smooth = cv2.GaussianBlur(row_mean.reshape(-1, 1),
                                  (1, ksize), 0).flatten()
    max_b = float(row_smooth.max())

    # ── 找所有显著"行间暗带"（面板行之间的黑缝）──
    # 用阈值法：连续低于 max_b * 0.55 的一段就是暗带
    th = max_b * 0.55
    n = len(row_smooth)
    is_bright = row_smooth > th

    # 找连续亮段
    segs = []
    start = None
    min_seg = max(100, int(n * 0.08))  # 至少占8%高度
    for i in range(n):
        if is_bright[i] and start is None:
            start = i
        elif not is_bright[i] and start is not None:
            if i - start >= min_seg:
                seg_mean = float(row_smooth[start:i].mean())
                segs.append((start, i, seg_mean, i - start))
            start = None
    if start is not None and n - start >= min_seg:
        seg_mean = float(row_smooth[start:n].mean())
        segs.append((start, n, seg_mean, n - start))

    if not segs:
        return y_top, y_bot

    # 选最长的亮段作为主导面板行
    segs.sort(key=lambda s: -s[3])
    best = segs[0]
    return y_top + best[0], y_top + best[1]


def _drone_find_panel_region(gray):
    """
    检测无人机图中主面板行的边界 (y_top, y_bot, x_left, x_right)。

    关键洞察：面板内部可能有薄的分隔线（如半片电池片的中间分隔），但：
    - 内部分隔线：很薄（<20px）且不太暗（minBrightness > 80）
    - 面板间间隙/边缘：较宽且很暗

    策略：
      1. Otsu 找整体亮区（rough x range）
      2. 在中间列做行投影
      3. 找所有"亮区带"（连续行的平均亮度 > 60%max）
      4. 若相邻两亮区间的暗缝很薄又浅（内部分隔线），合并它们
      5. 取合并后最高的带作为面板行
    """
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blur, max(otsu_val * 0.5, 15),
                            255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == largest)
    if len(xs) < 100:
        return None

    x_left = int(xs.min())
    x_right = int(xs.max())

    # ── 用行投影在整张图上找主面板行 ──
    x_mid_s = x_left + int((x_right - x_left) * 0.2)
    x_mid_e = x_left + int((x_right - x_left) * 0.8)
    strip = gray[:, x_mid_s:x_mid_e]
    row_mean = strip.astype(np.float32).mean(axis=1)
    ksize = max(21, int(len(row_mean) * 0.01)) | 1
    rs = cv2.GaussianBlur(row_mean.reshape(-1, 1),
                          (1, ksize), 0).flatten()
    max_b = float(rs.max())
    bright_th = max_b * 0.65

    # 找所有连续亮带
    bands = []
    in_b = False
    bs = 0
    for i in range(len(rs)):
        if rs[i] > bright_th and not in_b:
            bs = i; in_b = True
        elif rs[i] <= bright_th and in_b:
            bands.append([bs, i])
            in_b = False
    if in_b:
        bands.append([bs, len(rs)])

    if not bands:
        return (int(ys.min()), int(ys.max()), x_left, x_right)

    # 合并"薄而浅"的内部分隔（gap宽度 < 3%图高 且最暗 > 55%max）
    merged = [bands[0]]
    for b in bands[1:]:
        gap_start = merged[-1][1]
        gap_end = b[0]
        gap_w = gap_end - gap_start
        if gap_w == 0:
            merged[-1][1] = b[1]
            continue
        gap_min = float(rs[gap_start:gap_end].min())
        if gap_w < H * 0.03 and gap_min > max_b * 0.55:
            # 内部分隔线，合并
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))

    # 选最高的合并带作为主面板行
    merged = [m for m in merged if (m[1] - m[0]) > H * 0.25]
    if not merged:
        return (int(ys.min()), int(ys.max()), x_left, x_right)

    merged.sort(key=lambda b: -(b[1] - b[0]))
    best = merged[0]
    return (best[0], best[1], x_left, x_right)


def _drone_detect_n_panels(gray, region, max_panels=3):
    """
    自适应检测实际面板数量 (1/2/3) 并返回每个面板的 x 范围。
    策略：
      1. 用列亮度投影找波谷（面板间的暗缝）
      2. 用波谷将整体区域分成若干子区域
      3. 验证每个子区域的平均亮度 > 阈值 且宽度足够
    返回: panel_x_ranges = [(xl, xr), ...]
    """
    y_top, y_bot, x_left, x_right = region
    h = y_bot - y_top
    r_top = y_top + int(h * 0.2)
    r_bot = y_bot - int(h * 0.2)
    roi = gray[r_top:r_bot, x_left:x_right]
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        return []

    col_mean = roi.astype(np.float32).mean(axis=0)
    ksize = max(21, int(len(col_mean) * 0.02)) | 1
    col_smooth = cv2.GaussianBlur(col_mean.reshape(1, -1),
                                   (ksize, 1), 0).flatten()

    w = len(col_smooth)
    max_b = float(col_smooth.max())

    # ── 找所有显著波谷（面板间暗缝）──
    valleys = []
    depth_min = max_b * 0.08
    for i in range(20, w - 20):
        if col_smooth[i] < col_smooth[i - 15] and \
           col_smooth[i] < col_smooth[i + 15]:
            left_max = col_smooth[max(0, i - 200):i].max()
            right_max = col_smooth[i:min(w, i + 200)].max()
            depth = min(left_max, right_max) - col_smooth[i]
            if depth > depth_min:
                valleys.append((i, col_smooth[i], depth))

    # 按深度降序排序，然后去重（近距离保留最深的）
    valleys.sort(key=lambda v: -v[2])
    min_sep = max(80, int(w * 0.10))
    deduped = []
    for v in valleys:
        if all(abs(v[0] - sv[0]) > min_sep for sv in deduped):
            deduped.append(v)

    # 【修复】不再硬性只取 max_panels-1 个波谷，否则当画面里有"被画幅截断的
    # 半截面板"时，半截面板和相邻完整面板会被误合并成一段。改为：用所有
    # 显著波谷切分得到全部候选段，再按"宽度"+"是否贴画幅边缘"过滤掉半截面板。
    deduped.sort(key=lambda v: v[0])
    split_positions = [v[0] for v in deduped]

    W_img = gray.shape[1]
    edge_tol = max(4, int(W_img * 0.01))   # 距画幅边缘 1% 以内视为"贴边"

    # ── 构造候选子区域并验证 ──
    x_bounds = [0] + split_positions + [w]
    raw_segs = []
    for i in range(len(x_bounds) - 1):
        s, e = x_bounds[i], x_bounds[i + 1]
        seg_w = e - s
        if seg_w < w * 0.08:   # 过度窄段直接丢（<8% ROI 宽）
            continue
        region_mean = float(col_smooth[s:e].mean())
        if region_mean <= max_b * 0.55:
            continue
        abs_xl = x_left + s
        abs_xr = x_left + e
        on_left_edge  = abs_xl <= edge_tol
        on_right_edge = abs_xr >= W_img - edge_tol
        raw_segs.append({
            'xl': abs_xl, 'xr': abs_xr,
            'w': seg_w,
            'on_edge': on_left_edge or on_right_edge,
        })

    if not raw_segs:
        return []

    # 以非贴边段的中位宽度为"完整面板"基准，过滤贴边且明显偏窄的段
    inner_widths = [s['w'] for s in raw_segs if not s['on_edge']]
    if inner_widths:
        ref_w = float(np.median(inner_widths))
        kept = []
        for s in raw_segs:
            if s['on_edge'] and s['w'] < ref_w * 0.75:
                continue   # 贴画幅边缘且宽度不足 → 被切断的半截面板，丢弃
            kept.append(s)
        raw_segs = kept
    else:
        # 全部都贴边 → 至少丢掉宽度不足 15% 的（回退到原有逻辑）
        raw_segs = [s for s in raw_segs if s['w'] >= w * 0.15]

    candidates = [(s['xl'], s['xr']) for s in raw_segs]
    return candidates[:max_panels]


def _refine_panel_corners(edges, rough_pts, sw, sh):
    """
    使用 fitLine 对面板四边做亚像素级精确化。
    对每条边，在其附近的窄带内采集边缘点，拟合直线；四线求交得到更精确的角点。
    """
    pts = order_corners(rough_pts)

    def fit_edge_points(p1, p2):
        """在 p1→p2 连线附近的窄带内采集边缘点并拟合直线。"""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(np.hypot(dx, dy), 1e-6)
        # 单位向量（切向）
        tx, ty = dx / length, dy / length
        # 法向量
        nx, ny = -ty, tx
        margin_n = max(8, int(min(sw, sh) * 0.015))  # 法向 ±1.5%
        margin_t = length * 0.03                      # 切向延伸
        # 生成带状采样 mask
        ys, xs = np.where(edges > 0)
        if len(xs) < 10:
            return None
        ptsE = np.column_stack([xs.astype(np.float32),
                                ys.astype(np.float32)])
        rel = ptsE - np.array([p1[0], p1[1]], dtype=np.float32)
        d_n = rel[:, 0] * nx + rel[:, 1] * ny     # 法向距离
        d_t = rel[:, 0] * tx + rel[:, 1] * ty     # 切向距离
        sel = (np.abs(d_n) < margin_n) & \
              (d_t > -margin_t) & (d_t < length + margin_t)
        side_pts = ptsE[sel]
        if len(side_pts) < 10:
            return None
        line = cv2.fitLine(side_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # line = (vx, vy, x0, y0)
        return [float(line[0]), float(line[1]),
                float(line[2]), float(line[3])]

    # 四条边的方向
    edges_lines = [
        fit_edge_points(pts[0], pts[1]),  # top (TL→TR)
        fit_edge_points(pts[1], pts[2]),  # right (TR→BR)
        fit_edge_points(pts[2], pts[3]),  # bottom (BR→BL)
        fit_edge_points(pts[3], pts[0]),  # left (BL→TL)
    ]

    def line_intersect2(l1, l2):
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
    max_shift = min(sw, sh) * 0.05  # 精确化位移不得超过5%
    for i in range(4):
        new_pt = line_intersect2(edges_lines[(i - 1) % 4], edges_lines[i])
        if new_pt is not None:
            dist = np.hypot(new_pt[0] - pts[i][0], new_pt[1] - pts[i][1])
            if dist < max_shift:
                refined.append(new_pt)
                continue
        refined.append(pts[i])
    return np.array(refined, dtype=np.float32)


def _drone_detect_single_panel(gray_roi, pad_pct=0.005):
    """
    对无人机图中单个面板子区域做检测 — 严格以面板边框四角为基准。
    使用 Hough 直线检测找面板边框线，交点即角点。
    当面板边缘与图片边界重合时，自动使用图片边界作为虚拟边框线。

    v2 优化 (Bug6)：
    - 先用亮度掩膜定位真实的面板实心区域，边框线必须与该区域边界接近
    - 候选线按"与面板区域边界的吻合度"排序，避免选到内部栅线
    - pad_pct 默认为 0 时不做外扩（严格贴边）
    """
    sh, sw = gray_roi.shape[:2]

    def seg_to_line(x1, y1, x2, y2):
        vx, vy = float(x2 - x1), float(y2 - y1)
        ln = max(np.hypot(vx, vy), 1e-6)
        return [vx / ln, vy / ln, (x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def line_intersect(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        d = vx1 * vy2 - vy1 * vx2
        if abs(d) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    # ── 第0步：先用亮度找出面板实心区域作为锚点 ──
    blur_otsu = cv2.GaussianBlur(gray_roi, (25, 25), 0)
    otsu_val, _ = cv2.threshold(blur_otsu, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bright_mask = cv2.threshold(blur_otsu, max(otsu_val * 0.5, 15),
                                   255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k3, iterations=3)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, k3, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask)

    # 真实面板区域的边界（用于约束边框线选择）
    panel_top, panel_bot, panel_left, panel_right = 0, sh - 1, 0, sw - 1
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pl = stats[largest, cv2.CC_STAT_LEFT]
        pt = stats[largest, cv2.CC_STAT_TOP]
        pw = stats[largest, cv2.CC_STAT_WIDTH]
        ph = stats[largest, cv2.CC_STAT_HEIGHT]
        # 只在面积足够大时采信
        if pw * ph > sw * sh * 0.15:
            panel_left = pl
            panel_top = pt
            panel_right = pl + pw - 1
            panel_bot = pt + ph - 1

    # ── 方法1：Hough 直线法（必须贴近面板区域边界）──
    blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 80)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    min_ll = int(min(sw, sh) * 0.15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_ll, maxLineGap=15)

    top_line = bot_line = left_line = right_line = None

    if lines is not None and len(lines) >= 2:
        lines_arr = lines.reshape(-1, 4)
        h_lines, v_lines = [], []
        for x1, y1, x2, y2 in lines_arr:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)
            if angle < 20 or angle > 160:
                h_lines.append(((y1 + y2) / 2, length, x1, y1, x2, y2))
            elif 70 < angle < 110:
                v_lines.append(((x1 + x2) / 2, length, x1, y1, x2, y2))

        long_h = sw * 0.30    # 水平线至少30%宽度（排除内部栅线）
        long_v = sh * 0.20    # 竖直线至少20%高度

        tol = min(sw, sh) * 0.08  # 容差：距面板边界 8% 以内视为边框

        # 顶边：距 panel_top 最近的长水平线
        top_cands = [t for t in h_lines
                     if t[1] >= long_h and abs(t[0] - panel_top) < tol]
        if top_cands:
            top_line = min(top_cands, key=lambda t: abs(t[0] - panel_top))

        # 底边：距 panel_bot 最近的长水平线
        bot_cands = [t for t in h_lines
                     if t[1] >= long_h and abs(t[0] - panel_bot) < tol]
        if bot_cands:
            bot_line = min(bot_cands, key=lambda t: abs(t[0] - panel_bot))

        # 左边：距 panel_left 最近的长竖线
        left_cands = [t for t in v_lines
                      if t[1] >= long_v and abs(t[0] - panel_left) < tol]
        if left_cands:
            left_line = min(left_cands, key=lambda t: abs(t[0] - panel_left))

        # 右边：距 panel_right 最近的长竖线
        right_cands = [t for t in v_lines
                       if t[1] >= long_v and abs(t[0] - panel_right) < tol]
        if right_cands:
            right_line = min(right_cands, key=lambda t: abs(t[0] - panel_right))

    # ── 记录哪些边是 Hough 实际检测到的（完整性判断依据）──
    # 四边全由 Hough 找到 → 面板四角均在画面内 → is_complete=True
    all_hough_found = (top_line is not None and bot_line is not None and
                       left_line is not None and right_line is not None)

    # 对未找到的边框线，用面板区域/图片边界合成（仍用于尽量给出坐标）
    m = 2
    if top_line is None:
        y = max(m, panel_top)
        top_line = (y, sw, 0, y, sw - 1, y)
    if bot_line is None:
        y = min(sh - m, panel_bot)
        bot_line = (y, sw, 0, y, sw - 1, y)
    if left_line is None:
        x = max(m, panel_left)
        left_line = (x, sh, x, 0, x, sh - 1)
    if right_line is None:
        x = min(sw - m, panel_right)
        right_line = (x, sh, x, 0, x, sh - 1)

    # 四线求交
    if abs(top_line[0] - bot_line[0]) > sh * 0.3 and \
       abs(left_line[0] - right_line[0]) > sw * 0.3:
        tl = seg_to_line(*top_line[2:])
        bl = seg_to_line(*bot_line[2:])
        ll = seg_to_line(*left_line[2:])
        rl = seg_to_line(*right_line[2:])

        corners = [
            line_intersect(tl, ll),
            line_intersect(tl, rl),
            line_intersect(bl, rl),
            line_intersect(bl, ll),
        ]
        if all(c is not None for c in corners):
            pts = order_corners(np.array(corners, dtype=np.float32))
            w1 = np.linalg.norm(pts[1] - pts[0])
            h1 = np.linalg.norm(pts[3] - pts[0])
            ar = w1 / max(h1, 1e-6)
            if 0.15 < ar < 5.0:
                # Bug1: 使用 fitLine 对每条边做亚像素级精确化
                pts = _refine_panel_corners(edges, pts, sw, sh)
                # pad 外扩（上下 2x 左右）
                if pad_pct > 0:
                    pts = _expand_corners(pts, pad_pct)
                for i in range(4):
                    pts[i, 0] = np.clip(pts[i, 0], 0, sw - 1)
                    pts[i, 1] = np.clip(pts[i, 1], 0, sh - 1)
                return pts, all_hough_found

    # ── 方法2备用：亮度阈值法（Hough 完全失败时使用，视为不完整）──
    blur2 = cv2.GaussianBlur(gray_roi, (31, 31), 0)
    otsu_val, _ = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    best_pts, best_score = None, -1
    for frac in [1.0, 0.90, 0.82, 0.74, 0.66, 0.58, 0.50, 0.42, 0.35]:
        thresh = max(8, otsu_val * frac)
        _, mask = cv2.threshold(blur2, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ke, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ke, iterations=3)
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
        if area < sw * sh * 0.08 or area > sw * sh * 0.98:
            continue
        pts = _hull_to_quad(hull)
        o = order_corners(pts)
        w1 = np.linalg.norm(o[1] - o[0])
        w2 = np.linalg.norm(o[2] - o[3])
        h1 = np.linalg.norm(o[3] - o[0])
        h2 = np.linalg.norm(o[2] - o[1])
        ar = max(w1, w2) / max(max(h1, h2), 1e-6)
        if ar < 0.2 or ar > 4.0:
            continue
        rect = (min(w1, w2) / max(w1, w2)) * (min(h1, h2) / max(h1, h2))
        score = area * rect
        if score > best_score:
            best_score, best_pts = score, pts

    if best_pts is None:
        return None, False

    pts = order_corners(best_pts)
    pts = _expand_corners(pts, pad_pct)
    for i in range(4):
        pts[i, 0] = np.clip(pts[i, 0], 0, sw - 1)
        pts[i, 1] = np.clip(pts[i, 1], 0, sh - 1)
    # 方法2（亮度法）无法区分单边是否被裁断，一律视为不完整
    return pts, False


def _panel_is_complete(pts_full, W, H, margin_frac=0.02):
    """
    判断面板四个角点是否全部处于图像内部（四角可见，未被画幅裁切）。

    原理：
      检测算法在遇到图像边界时会将角点 clip 到边缘坐标，
      导致贴边角点的 x 或 y 值恰好等于 0 / W-1 / H-1。
      只要有一个角点距图像任意边缘不足 margin_frac 比例，
      就认定该面板在此方向超出画幅，属于不完整面板，不予裁剪。

    参数：
      margin_frac : 判定阈值，以图像宽/高的比例计算（默认 2%）。
                    考虑到 pad 外扩最多约 3% 面板高度，2% 足以
                    区分"正常边距"与"被画幅截断"两种情况。
    """
    margin_x = W * margin_frac
    margin_y = H * margin_frac
    for x, y in pts_full:
        if x < margin_x or x > W - margin_x:
            return False   # 左右方向被截断
        if y < margin_y or y > H - margin_y:
            return False   # 上下方向被截断
    return True


def process_drone_image(img_bytes, pad_pct=0.014, v_pad=None):
    """
    无人机拍摄模式：自适应检测并裁出 1/2/3 张面板（左/中/右）。
    角点坐标为全图坐标。输出统一尺寸。
    仅裁剪四角完整可见的面板，超出画幅的面板自动跳过。
    """
    if v_pad is None:
        v_pad = pad_pct  # 默认与水平一致
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, 'Decode failed'
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        region = _drone_find_panel_region(gray)
        if region is None:
            return None, 'Panel region not detected'
        y_top, y_bot, x_left, x_right = region

        # 自适应检测面板数量
        panel_ranges = _drone_detect_n_panels(gray, region, max_panels=3)
        if not panel_ranges:
            return None, 'No panels detected'

        n_panels = len(panel_ranges)
        # 左/中/右 的定义：从左开始依次分配
        #   1 张 → 左
        #   2 张 → 左、中
        #   3 张 → 左、中、右
        all_labels = ['左', '中', '右'][:n_panels]

        # ── 第1遍：检测各面板角点（全图坐标），跳过四角不完整的面板 ──
        # 完整性标准：四个角点全部远离图像边缘（距任意边 ≥ 图像宽/高的 2%）
        valid_panels = []   # 每项: (原始pi, label, pts_full)
        skipped = []

        for pi in range(n_panels):
            xl_p, xr_p = panel_ranges[pi]
            pw = xr_p - xl_p
            # ROI 边界固定为 3% 面板宽（检测用，外扩在检测后单独做）
            margin = int(pw * 0.03)

            # 外侧面板允许扩展到图像边缘，内部面板用相邻段中点约束
            if pi > 0:
                prev_xr = panel_ranges[pi - 1][1]
                left_hard = (prev_xr + xl_p) // 2
            else:
                left_hard = 0   # 最左侧面板自由扩展到左边缘

            if pi < n_panels - 1:
                next_xl = panel_ranges[pi + 1][0]
                right_hard = (xr_p + next_xl + 1) // 2
            else:
                right_hard = W  # 最右侧面板自由扩展到右边缘

            xl = max(left_hard, xl_p - margin)
            xr = min(right_hard, xr_p + margin)
            # 垂直方向 ROI 边距（固定 3%）
            margin_y = int(pw * 0.03)
            yt = max(0, y_top - margin_y)
            yb = min(H, y_bot + margin_y)

            label = all_labels[pi] if pi < len(all_labels) else str(pi)

            gray_roi = gray[yt:yb, xl:xr]
            # 检测时 pad=0（不做内部外扩），统一在外部做不对称外扩
            pts_roi, hough_complete = _drone_detect_single_panel(gray_roi, 0)

            if pts_roi is None:
                skipped.append(label + '(检测失败)')
                print('  [无人机] 跳过%s面板：角点检测失败' % label)
                continue

            # 转换为全图坐标
            pts_full = pts_roi.copy()
            pts_full[:, 0] += xl
            pts_full[:, 1] += yt

            # 相邻面板间容差：1% 面板宽，保证不互相吃进又保留边框
            neighbor_tol = pw * 0.01
            if pi > 0:
                pts_full[0, 0] = max(pts_full[0, 0], xl_p - neighbor_tol)
                pts_full[3, 0] = max(pts_full[3, 0], xl_p - neighbor_tol)
            if pi < n_panels - 1:
                pts_full[1, 0] = min(pts_full[1, 0], xr_p + neighbor_tol)
                pts_full[2, 0] = min(pts_full[2, 0], xr_p + neighbor_tol)

            # ── 不对称外扩：水平/垂直分别用 h_pad / v_pad ──
            if pad_pct > 0:
                h_side = max(np.linalg.norm(pts_full[3] - pts_full[0]),
                              np.linalg.norm(pts_full[2] - pts_full[1]))
                cx, cy = pts_full.mean(axis=0)
                expand_h = round(h_side * pad_pct)
                expand_v = round(h_side * v_pad)
                for j in range(4):
                    dx, dy = pts_full[j] - np.array([cx, cy])
                    d = max(np.hypot(dx, dy), 1e-6)
                    pts_full[j] += np.array([dx/d * expand_h, dy/d * expand_v])
                pts_full[:, 0] = np.clip(pts_full[:, 0], 0, W - 1)
                pts_full[:, 1] = np.clip(pts_full[:, 1], 0, H - 1)

            # ── 完整性检验：有 padding 时角点自然靠近边缘，跳过检查
            # 无 padding 时用 margin_frac 验证角点是否在画幅内
            if pad_pct <= 0:
                if not _panel_is_complete(pts_full, W, H, margin_frac=0.02):
                    skipped.append(label + '(超出画幅)')
                    print('  [无人机] 跳过%s面板：角点贴近图像边缘，判定为不完整面板' % label)
                    continue

            valid_panels.append((pi, label, pts_full))

        if not valid_panels:
            reason = '、'.join(skipped) if skipped else '无候选'
            return None, '未检测到完整面板（' + reason + '）'

        if skipped:
            print('  [无人机] 已跳过 %d 个不完整面板（%s），输出 %d 张'
                  % (len(skipped), '、'.join(skipped), len(valid_panels)))

        panel_corners_list = [pts for _, _, pts in valid_panels]

        # 统一输出尺寸（取完整面板的中位数）
        widths, heights = [], []
        for pts in panel_corners_list:
            w = int(max(np.linalg.norm(pts[1] - pts[0]),
                        np.linalg.norm(pts[2] - pts[3])))
            h = int(max(np.linalg.norm(pts[3] - pts[0]),
                        np.linalg.norm(pts[2] - pts[1])))
            widths.append(w)
            heights.append(h)
        target_w = max(int(np.median(widths)), 10)
        target_h = max(int(np.median(heights)), 10)

        # 第2遍：透视变换（只处理完整面板）
        results = []
        for pi, label, pts in valid_panels:
            pts = pts.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            dst = np.array([[0, 0], [target_w, 0],
                            [target_w, target_h], [0, target_h]],
                           dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(img, M, (target_w, target_h),
                                         borderMode=cv2.BORDER_REPLICATE)
            ok, buf = cv2.imencode('.jpg', warped,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None, 'Encode failed for panel ' + label
            results.append((buf.tobytes(), label,
                            pts.tolist(), None))

        meta = {
            'region': list(region),
            'panel_ranges': panel_ranges,
            'n_panels': len(valid_panels),
            'skipped': skipped,
            'img_w': W, 'img_h': H,
            'target_w': target_w, 'target_h': target_h,
        }
        return results, meta

    except Exception as e:
        return None, str(e)


def process_drone_single(img_bytes, corners_full, target_w=0, target_h=0):
    """
    重新裁剪无人机图中的单个面板。
    corners_full: 全图坐标的4个角点 [[x,y],...] (TL,TR,BR,BL)
    target_w/target_h: 输出尺寸（0=自动）
    """
    try:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, 'Decode failed'
        H, W = img.shape[:2]
        pts = np.array(corners_full, dtype=np.float32)

        # Bug2修复：严格限制角点在图片范围内
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        if target_w <= 0 or target_h <= 0:
            target_w = int(max(np.linalg.norm(pts[1] - pts[0]),
                               np.linalg.norm(pts[2] - pts[3])))
            target_h = int(max(np.linalg.norm(pts[3] - pts[0]),
                               np.linalg.norm(pts[2] - pts[1])))

        dst = np.array([[0, 0], [target_w, 0],
                        [target_w, target_h], [0, target_h]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        # 用 BORDER_REPLICATE 避免采样到图像外时出现白/黑填充
        warped = cv2.warpPerspective(img, M, (target_w, target_h),
                                     borderMode=cv2.BORDER_REPLICATE)

        ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None, 'Encode failed'
        return {'jpeg': buf.tobytes(), 'corners': pts.tolist()}, None
    except Exception as e:
        return None, str(e)


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

def _load_corrections(path=None):
    p = path or CORR_F
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_corrections(data, path=None):
    p = path or CORR_F
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_drone_corrections():
    return _load_corrections(CORR_DRONE_F)

def _save_drone_corrections(data):
    _save_corrections(data, CORR_DRONE_F)

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

@app.route('/logo')
def logo():
    """返回 app 文件夹内的 logo.ico"""
    logo_path = os.path.join(BASE, 'logo.ico')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            return Response(f.read(), mimetype='image/x-icon')
    return Response(b'', status=404)

@app.route('/health')
def health():
    return jsonify({
        'ok': True,
        'backend': True,
        'frontend': os.path.exists(HTML),
        'authenticated': session.get('logged_in') is True,
        'user': session.get('user', ''),
        'port': 15789,
    })

@app.route('/auth/status')
def auth_status():
    return jsonify({
        'ok': True,
        'authenticated': session.get('logged_in') is True,
        'user': session.get('user', ''),
        'role': session.get('role', ''),
    })

@app.route('/auth/login', methods=['POST'])
def auth_login():
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    password = str(d.get('password', ''))

    if _validate_user(username, password):
        session['logged_in'] = True
        session['user'] = username
        session['role'] = _user_role(username)
        return jsonify({
            'ok': True, 'authenticated': True,
            'user': username,
            'role': session['role']
        })
    return jsonify({'ok': False, 'authenticated': False, 'error': '用户名或密码错误'}), 401

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/auth/check-username', methods=['POST'])
def auth_check_username():
    """Check if username exists (for account recovery)"""
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    users = _load_users()
    exists = any(u['username'] == username for u in users)
    return jsonify({'ok': True, 'exists': exists})

@app.route('/auth/recover', methods=['POST'])
def auth_recover():
    """Recover account - verify known credentials and return user info"""
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    users = _load_users()
    for u in users:
        if u['username'] == username:
            return jsonify({
                'ok': True,
                'username': u['username'],
                'role': u.get('role', 'user'),
                'created_at': u.get('created_at', '')
            })
    return jsonify({'ok': False, 'error': '账户不存在'}), 404

@app.route('/auth/users', methods=['GET'])
def auth_list_users():
    """List all users (admin only)"""
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '需要管理员权限'}), 403
    users = _load_users()
    safe = [{'username': u['username'], 'role': u.get('role', 'user'),
             'created_at': u.get('created_at', '')} for u in users]
    return jsonify({'ok': True, 'users': safe})

@app.route('/auth/users/add', methods=['POST'])
def auth_add_user():
    """Add a new user (admin only)"""
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '需要管理员权限'}), 403
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    password = str(d.get('password', ''))
    role = str(d.get('role', 'user'))
    ok, msg = _add_user(username, password, role)
    if ok:
        return jsonify({'ok': True, 'message': msg})
    return jsonify({'ok': False, 'error': msg}), 400

@app.route('/auth/users/delete', methods=['POST'])
def auth_delete_user():
    """Delete a user (admin only, cannot delete self)"""
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '需要管理员权限'}), 403
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    current = session.get('user', '')
    if username == current:
        return jsonify({'ok': False, 'error': '不能删除自己'}), 400
    _remove_user(username)
    return jsonify({'ok': True, 'message': f'用户 {username} 已删除'})

@app.route('/auth/users/password', methods=['POST'])
def auth_change_user_password():
    """Change user password (admin only)"""
    if session.get('role') != 'admin':
        return jsonify({'ok': False, 'error': '需要管理员权限'}), 403
    try:
        d = request.get_json(force=True)
    except Exception:
        d = {}
    username = str(d.get('username', '')).strip()
    new_password = str(d.get('password', ''))
    if len(new_password) < 4:
        return jsonify({'ok': False, 'error': '密码至少4个字符'}), 400
    if _change_password(username, new_password):
        return jsonify({'ok': True, 'message': '密码已修改'})
    return jsonify({'ok': False, 'error': '用户不存在'}), 404

@app.route('/info')
def info():
    dev = ''
    if USE_GPU:
        try: dev = cv2.ocl.Device.getDefault().name()
        except: dev = 'OpenCL'
    with _lock:
        corr = _load_corrections()
        drone_corr = _load_drone_corrections()
    return jsonify({'gpu': USE_GPU, 'workers': WORKERS,
                    'cv': cv2.__version__, 'device': dev,
                    'corrections': len(corr),
                    'corrections_drone': len(drone_corr)})

@app.route('/process', methods=['POST'])
def process():
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f: return jsonify({'ok': False, 'error': 'No image'})
        pad     = float(request.form.get('pad', 0.005))
        pad_v   = float(request.form.get('pad_v', pad))  # 上下外扩，默认与左右一致
        forced  = json.loads(request.form['corners']) \
                  if request.form.get('corners') else None
        img_name = request.form.get('name', None)
        jpeg, corners, err = process_image(f.read(), pad, forced, img_name, pad_v)
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
    pad = float(request.form.get('pad', 0.005))
    pad_v = float(request.form.get('pad_v', pad))
    forced = json.loads(request.form['corners']) \
             if request.form.get('corners') else None
    names_raw = request.form.get('names', None)
    names = json.loads(names_raw) if names_raw else [None] * len(fs)
    if len(names) != len(fs):
        names = [None] * len(fs)

    def proc(f, name):
        return process_image(f.read(), pad, forced, name, pad_v)

    futures = {pool.submit(proc, f, n): i for i, (f, n) in enumerate(zip(fs, names))}
    results = [None] * len(fs)
    for fut, idx in futures.items():
        jpeg, corners, err = fut.result()
        results[idx] = {'ok': False, 'error': err} if err else \
                       {'ok': True,
                        'jpeg_b64': base64.b64encode(jpeg).decode(),
                        'corners': corners}
    return jsonify({'results': results, 'ms': round((time.time()-t0)*1000)})

@app.route('/process_drone', methods=['POST'])
def process_drone():
    """无人机拍摄模式：一张图 → 三张裁剪结果（左/中/右）"""
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f:
            return jsonify({'ok': False, 'error': 'No image'})
        pad = float(request.form.get('pad', 0.014))
        pad_v = float(request.form.get('pad_v', pad))
        result, meta_or_err = process_drone_image(f.read(), pad, pad_v)
        if isinstance(meta_or_err, str):
            return jsonify({'ok': False, 'error': meta_or_err})
        panels = []
        for jpeg_bytes, label, corners, roi_bounds in result:
            panels.append({
                'label': label,
                'jpeg_b64': base64.b64encode(jpeg_bytes).decode(),
                'corners': corners,
                'roi': roi_bounds,
            })
        return jsonify({
            'ok': True,
            'panels': panels,
            'meta': meta_or_err,
            'ms': round((time.time() - t0) * 1000)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})

@app.route('/process_drone_single', methods=['POST'])
def process_drone_single_route():
    """无人机模式：用全图角点重新裁剪单个面板"""
    t0 = time.time()
    try:
        f = request.files.get('image')
        if not f:
            return jsonify({'ok': False, 'error': 'No image'})
        corners = json.loads(request.form['corners'])
        tw = int(request.form.get('target_w', 0))
        th = int(request.form.get('target_h', 0))
        result, err = process_drone_single(f.read(), corners, tw, th)
        if err:
            return jsonify({'ok': False, 'error': err})
        return jsonify({
            'ok': True,
            'jpeg_b64': base64.b64encode(result['jpeg']).decode(),
            'corners': result['corners'],
            'ms': round((time.time() - t0) * 1000)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': traceback.format_exc()})

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
            'pad_v':  float(d.get('pad_v', d.get('pad', 0.03))),
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
    _update_learned_bias()
    return jsonify({'ok': True})

# ── Fix1：列出所有学习样本（供前端展示可删除列表）──
@app.route('/corrections/list')
def list_corrections():
    mode = request.args.get('mode', 'normal')
    with _lock:
        corr = _load_drone_corrections() if mode == 'drone' else _load_corrections()
    items = []
    for c in corr:
        auto = np.array(c.get('auto', []), dtype=np.float32)
        manual = np.array(c.get('manual', []), dtype=np.float32)
        if auto.size and manual.size and auto.shape == manual.shape:
            mean_err = float(np.abs(manual - auto).mean())
        else:
            mean_err = 0.0
        items.append({
            'name': c.get('name', '?'),
            'img_w': c.get('img_w', 0),
            'img_h': c.get('img_h', 0),
            'pad': c.get('pad', 0.03),
            'ts': c.get('ts', 0),
            'mean_err': round(mean_err, 1),
            'panel': c.get('panel', ''),   # 无人机才有
        })
    items.sort(key=lambda x: -x['ts'])
    return jsonify({'ok': True, 'items': items, 'mode': mode})

# ── Fix1：删除单个学习样本 ──
@app.route('/correction/delete', methods=['POST'])
def delete_correction():
    try:
        d = request.get_json(force=True)
        name = d.get('name', '')
        panel = d.get('panel', '')   # 无人机模式下需要指定哪个面板
        mode = d.get('mode', 'normal')
        if not name:
            return jsonify({'ok': False, 'error': 'name required'})
        with _lock:
            if mode == 'drone':
                corr = _load_drone_corrections()
                before = len(corr)
                if panel:
                    corr = [c for c in corr
                            if not (c.get('name') == name and c.get('panel') == panel)]
                else:
                    corr = [c for c in corr if c.get('name') != name]
                _save_drone_corrections(corr)
            else:
                corr = _load_corrections()
                before = len(corr)
                corr = [c for c in corr if c.get('name') != name]
                _save_corrections(corr)
            removed = before - len(corr)
        if mode == 'normal':
            _update_learned_bias()
        return jsonify({'ok': True, 'removed': removed, 'total': len(corr)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

# ── Fix5：无人机模式独立的学习样本保存/分析/清空 ──
@app.route('/correction_drone', methods=['POST'])
def save_correction_drone():
    """无人机模式的单面板修正记录（与普通模式完全隔离）"""
    try:
        d = request.get_json(force=True)
        auto = np.array(d['auto'], dtype=np.float32)
        manual = np.array(d['manual'], dtype=np.float32)
        record = {
            'name':   d.get('name', 'unknown'),
            'panel':  d.get('panel', '左'),    # 左/中/右
            'img_w':  int(d.get('img_w', 0)),
            'img_h':  int(d.get('img_h', 0)),
            'pad':    float(d.get('pad', 0.03)),
            'pad_v':  float(d.get('pad_v', d.get('pad', 0.03))),
            'auto':   auto.tolist(),
            'manual': manual.tolist(),
            'ts':     int(time.time()),
        }
        with _lock:
            corr = _load_drone_corrections()
            # 同一图同一面板只保留最新记录
            corr = [c for c in corr
                    if not (c.get('name') == record['name']
                            and c.get('panel') == record['panel'])]
            corr.append(record)
            _save_drone_corrections(corr)
        return jsonify({'ok': True, 'saved': True, 'total': len(corr)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/analysis_drone')
def analysis_drone():
    try:
        with _lock:
            corr = _load_drone_corrections()
        result = _compute_analysis(corr)
        if result is None:
            return jsonify({'ok': True, 'count': 0, 'ready': False})
        # 统计每个面板的样本数
        panel_counts = {'左': 0, '中': 0, '右': 0}
        for c in corr:
            p = c.get('panel', '')
            if p in panel_counts:
                panel_counts[p] += 1
        result['ok'] = True
        result['ready'] = True
        result['panel_counts'] = panel_counts
        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/corrections_drone/clear', methods=['POST'])
def clear_corrections_drone():
    with _lock:
        _save_drone_corrections([])
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


if __name__ == '__main__':
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
