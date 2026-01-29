import io
import os
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from PIL import Image

app = Flask(__name__)

# -----------------------------
# Quilting settings
# -----------------------------
PATCH = 220
OVERLAP = 60

def min_cut_path(cost):
    h, w = cost.shape
    dp = cost.copy()
    back = np.zeros_like(dp, dtype=np.int32)

    for i in range(1, h):
        for j in range(w):
            prev_js = [j]
            if j > 0: prev_js.append(j - 1)
            if j < w - 1: prev_js.append(j + 1)
            prev_vals = [dp[i - 1, pj] for pj in prev_js]
            best_idx = int(np.argmin(prev_vals))
            back[i, j] = prev_js[best_idx]
            dp[i, j] += prev_vals[best_idx]

    cut = np.zeros(h, dtype=np.int32)
    cut[h - 1] = int(np.argmin(dp[h - 1]))
    for i in range(h - 2, -1, -1):
        cut[i] = back[i + 1, cut[i + 1]]
    return cut

def quilt_texture(tex, out_h, out_w, patch=PATCH, overlap=OVERLAP):
    tex_h, tex_w = tex.shape[:2]
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    step = patch - overlap
    rng = np.random.default_rng()

    for y in range(0, out_h, step):
        for x in range(0, out_w, step):
            ry = int(rng.integers(0, max(1, tex_h - patch)))
            rx = int(rng.integers(0, max(1, tex_w - patch)))
            p = tex[ry:ry+patch, rx:rx+patch].copy()

            y2 = min(y + patch, out_h)
            x2 = min(x + patch, out_w)
            p = p[:y2 - y, :x2 - x]

            if x == 0 and y == 0:
                canvas[y:y2, x:x2] = p
                continue

            region = canvas[y:y2, x:x2]

            # left seam
            if x > 0:
                ov = min(overlap, x2 - x)
                left_existing = region[:, :ov].astype(np.float32)
                left_new = p[:, :ov].astype(np.float32)
                cost = np.sum((left_existing - left_new) ** 2, axis=2)
                cut = min_cut_path(cost)

                mask = np.zeros((p.shape[0], ov), dtype=np.uint8)
                for i in range(p.shape[0]):
                    mask[i, :cut[i]] = 255
                mask3 = cv2.merge([mask, mask, mask])
                blended_left = np.where(mask3 == 255, left_existing, left_new).astype(np.uint8)
                p[:, :ov] = blended_left

            # top seam
            if y > 0:
                ov = min(overlap, y2 - y)
                top_existing = region[:ov, :].astype(np.float32)
                top_new = p[:ov, :].astype(np.float32)
                cost = np.sum((top_existing - top_new) ** 2, axis=2).T
                cut = min_cut_path(cost)

                mask = np.zeros((ov, p.shape[1]), dtype=np.uint8)
                for j in range(p.shape[1]):
                    mask[:cut[j], j] = 255
                mask3 = cv2.merge([mask, mask, mask])
                blended_top = np.where(mask3 == 255, top_existing, top_new).astype(np.uint8)
                p[:ov, :] = blended_top

            canvas[y:y2, x:x2] = p

    return canvas

def read_image_file(file_storage):
    # supports PNG/JPG, keeps alpha if present
    data = file_storage.read()
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    arr = np.array(img)  # RGBA
    return arr

def rgba_to_bgr(rgba):
    rgb = rgba[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def read_mask_file(mask_storage, target_h, target_w):
    data = mask_storage.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    m = np.array(img)
    m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def apply_fabric(bed_bgr, fabric_bgr, mask):
    h, w = bed_bgr.shape[:2]
    big = quilt_texture(fabric_bgr, h, w)

    bed_gray = cv2.cvtColor(bed_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m = (mask.astype(np.float32) / 255.0)
    eps = 1e-6
    mean_light = (bed_gray * m).sum() / (m.sum() + eps)
    light_map = np.clip((bed_gray / (mean_light + eps)), 0.55, 1.5)

    big_f = big.astype(np.float32) / 255.0
    lit = np.clip(big_f * light_map[..., None], 0, 1)
    lit = (lit * 255).astype(np.uint8)

    # soft edge
    mask_soft = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
    out = bed_bgr.astype(np.float32) * (1 - mask_soft[..., None]) + lit.astype(np.float32) * mask_soft[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.post("/apply")
def apply_endpoint():
    if "bed" not in request.files or "fabric" not in request.files or "mask" not in request.files:
        return jsonify(error="Send multipart/form-data with bed, fabric, mask"), 400

    bed_rgba = read_image_file(request.files["bed"])
    fabric_rgba = read_image_file(request.files["fabric"])
    mask = read_mask_file(request.files["mask"], bed_rgba.shape[0], bed_rgba.shape[1])

    bed_bgr = rgba_to_bgr(bed_rgba)
    fabric_bgr = rgba_to_bgr(fabric_rgba)

    out_bgr = apply_fabric(bed_bgr, fabric_bgr, mask)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # keep original alpha from bed (so background stays transparent if bed PNG had it)
    alpha = bed_rgba[..., 3:4]
    out_rgba = np.concatenate([out_rgb, alpha], axis=2)

    out_pil = Image.fromarray(out_rgba, mode="RGBA")
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="bed_fabric.png")
