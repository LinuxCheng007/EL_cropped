# 批量无人机模式裁剪脚本（独立版本，不依赖 importlib）
import sys, os, json, glob

# ── 添加 runtime site-packages ──
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET = os.path.join(_ROOT, 'runtime', 'Lib', 'site-packages')
if os.path.isdir(_TARGET) and _TARGET not in sys.path:
    sys.path.insert(0, _TARGET)
_PYDIR = os.path.join(_ROOT, 'runtime')
for _sp in ['Lib\\site-packages', 'site-packages']:
    _p = os.path.join(_PYDIR, _sp)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np
import cv2
import importlib.util

# ── 导入 app.py（已确保 path 正确）──
spec = importlib.util.spec_from_file_location("crop_app", os.path.join(_APP_DIR, "app.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
process_drone_image = mod.process_drone_image

# ── 配置 ──
SRC_DIR = r"C:\Users\Linux\Desktop\分类\图片"
OUT_DIR = os.path.join(SRC_DIR, "结果")
PAD_PCT = 0.008

os.makedirs(OUT_DIR, exist_ok=True)

# ── 收集图片 ──
exts = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG')
files = []
for ext in exts:
    files.extend(glob.glob(os.path.join(SRC_DIR, ext)))
files = sorted(set(files))

print("=" * 60)
print("  无人机面板裁剪 批量处理 (pid=%d)" % os.getpid())
print("  源目录: %s" % SRC_DIR)
print("  输出目录: %s" % OUT_DIR)
print("  图片总数: %d 张" % len(files))
print("=" * 60)

total_ok = 0
total_fail = 0
summary = []

for fi, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    name_noext = os.path.splitext(fname)[0]
    out_sub = os.path.join(OUT_DIR, name_noext)
    os.makedirs(out_sub, exist_ok=True)

    sys.stdout.write("\n[%d/%d] %s" % (fi + 1, len(files), fname))
    sys.stdout.flush()

    try:
        with open(fpath, 'rb') as f:
            img_bytes = f.read()

        results, meta_or_err = process_drone_image(img_bytes, PAD_PCT)

        if isinstance(meta_or_err, str):
            sys.stdout.write("  FAILED: %s\n" % meta_or_err)
            total_fail += 1
            summary.append((fname, 'FAIL', meta_or_err))
            continue

        panels = results
        for jpeg_bytes, label, corners, _ in panels:
            out_name = "%s_%s.jpg" % (name_noext, label)
            out_path = os.path.join(out_sub, out_name)
            with open(out_path, 'wb') as f:
                f.write(jpeg_bytes)
            sys.stdout.write("  [%s]" % label)

        sys.stdout.write("  OK (%d panels)\n" % len(panels))
        total_ok += 1
        summary.append((fname, 'OK', '%d panels' % len(panels)))

    except Exception as e:
        import traceback
        sys.stdout.write("  ERROR: %s\n" % traceback.format_exc())
        total_fail += 1
        summary.append((fname, 'ERROR', str(e)))

print("\n" + "=" * 60)
print("  处理完成!")
print("  成功: %d 张" % total_ok)
print("  失败: %d 张" % total_fail)
print("=" * 60)

# 写入报告
report_path = os.path.join(OUT_DIR, '_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total': len(files),
        'ok': total_ok,
        'fail': total_fail,
        'summary': [{'file': s[0], 'status': s[1], 'detail': s[2]} for s in summary],
    }, f, ensure_ascii=False, indent=2)
print("\n报告已保存: %s" % report_path)

# 验证文件写入
n_written = sum(1 for root, dirs, fnames in os.walk(OUT_DIR) for _ in fnames)
print("  输出目录文件数: %d" % (n_written - 1))  # 减掉 _report.json
print("按 Enter 退出...")
try:
    input()
except:
    pass
