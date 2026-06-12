"""颜色调整模块：白平衡校正、色调曲线、HSL调节、LUT滤镜"""

from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
from PIL import Image
import cv2
from scipy import interpolate

from .core import pil_to_cv, cv_to_pil


def adjust_white_balance_auto(image: Image.Image) -> Image.Image:
    """自动白平衡校正（灰度世界算法 + 简化白点检测）"""
    cv_img = pil_to_cv(image)
    
    result = _gray_world_wb(cv_img)
    
    h, w = cv_img.shape[:2]
    small = cv2.resize(cv_img, (min(w, 200), min(h, 200)))
    max_percent = 0.005
    pixels = small.reshape(-1, 3).astype(np.float32)
    brightness = np.mean(pixels, axis=1)
    n_brightest = max(1, int(len(pixels) * max_percent))
    brightest_idx = np.argsort(brightness)[-n_brightest:]
    white_pixels = pixels[brightest_idx]
    if len(white_pixels) > 0 and np.mean(white_pixels) > 200:
        avg_white = np.mean(white_pixels, axis=0)
        scale = 255.0 / np.maximum(avg_white, 1.0)
        result_wp = cv_img.astype(np.float32) * scale.reshape(1, 1, 3)
        result_wp = np.clip(result_wp, 0, 255).astype(np.uint8)
        result = cv2.addWeighted(result, 0.7, result_wp, 0.3, 0)
    
    return cv_to_pil(result)


def _gray_world_wb(cv_img: np.ndarray) -> np.ndarray:
    """灰度世界白平衡算法"""
    result = cv_img.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    scale_b = avg_gray / max(avg_b, 1.0)
    scale_g = avg_gray / max(avg_g, 1.0)
    scale_r = avg_gray / max(avg_r, 1.0)
    
    max_scale = 2.5
    scale_b = min(scale_b, max_scale)
    scale_g = min(scale_g, max_scale)
    scale_r = min(scale_r, max_scale)
    
    result[:, :, 0] *= scale_b
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_r
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_temperature_tint(
    image: Image.Image,
    temperature: float = 0.0,
    tint: float = 0.0,
) -> Image.Image:
    """调整色温(冷暖)和色调(绿品)
    
    Args:
        temperature: -100 (冷蓝) 到 100 (暖黄)
        tint: -100 (偏绿) 到 100 (偏品红)
    """
    cv_img = pil_to_cv(image).astype(np.float32)
    
    temp_factor = temperature / 100.0
    tint_factor = tint / 100.0
    
    r_gain = 1.0 + temp_factor * 0.3
    b_gain = 1.0 - temp_factor * 0.3
    g_gain_tint = 1.0 - tint_factor * 0.2
    r_gain_tint = 1.0 + tint_factor * 0.15
    b_gain_tint = 1.0 + tint_factor * 0.15
    
    cv_img[:, :, 2] *= r_gain * r_gain_tint
    cv_img[:, :, 1] *= g_gain_tint
    cv_img[:, :, 0] *= b_gain * b_gain_tint
    
    cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
    return cv_to_pil(cv_img)


def adjust_curves(
    image: Image.Image,
    channel: str = "rgb",
    points: Optional[List[Tuple[int, int]]] = None,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> Image.Image:
    """色调曲线调整
    
    Args:
        channel: 调整通道 - rgb, r, g, b
        points: 控制点列表 [(输入值, 输出值), ...]，如 [(0,0), (128,140), (255,255)]
        brightness: -100 到 100
        contrast: -100 到 100
    """
    cv_img = pil_to_cv(image)
    
    if points is None:
        points = [(0, 0), (255, 255)]
    
    points = sorted(points, key=lambda p: p[0])
    if points[0][0] != 0:
        points.insert(0, (0, 0))
    if points[-1][0] != 255:
        points.append((255, 255))
    
    x_in = [p[0] for p in points]
    y_out = [p[1] for p in points]
    
    lut = np.arange(256, dtype=np.uint8)
    if len(x_in) >= 2:
        tck = interpolate.interp1d(
            x_in, y_out, kind="cubic" if len(x_in) > 3 else "linear",
            fill_value="extrapolate"
        )
        lut = np.clip(tck(np.arange(256)), 0, 255).astype(np.uint8)
    
    if brightness != 0 or contrast != 0:
        base_lut = np.arange(256, dtype=np.float32)
        if contrast != 0:
            contrast_factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
            base_lut = contrast_factor * (base_lut - 128.0) + 128.0
        if brightness != 0:
            base_lut += brightness * 2.55
        base_lut = np.clip(base_lut, 0, 255).astype(np.uint8)
        lut = base_lut[lut]
    
    result = cv_img.copy()
    channel = channel.lower()
    
    if channel == "rgb":
        for c in range(3):
            result[:, :, c] = cv2.LUT(cv_img[:, :, c], lut)
    elif channel == "b":
        result[:, :, 0] = cv2.LUT(cv_img[:, :, 0], lut)
    elif channel == "g":
        result[:, :, 1] = cv2.LUT(cv_img[:, :, 1], lut)
    elif channel == "r":
        result[:, :, 2] = cv2.LUT(cv_img[:, :, 2], lut)
    
    return cv_to_pil(result)


def adjust_hsl(
    image: Image.Image,
    hue: float = 0.0,
    saturation: float = 0.0,
    lightness: float = 0.0,
    hue_range: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """HSL单独调节
    
    Args:
        hue: -180 到 180 (色相偏移)
        saturation: -100 到 100
        lightness: -100 到 100
        hue_range: 指定色相范围 (min_hue, max_hue) 0-360，None表示全局调整
    """
    cv_img = pil_to_cv(image)
    hls = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HLS).astype(np.float32)
    
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    
    if hue_range is not None:
        h_min, h_max = hue_range
        if h_min <= h_max:
            mask = ((h >= h_min) & (h <= h_max)).astype(np.float32)
        else:
            mask = ((h >= h_min) | (h <= h_max)).astype(np.float32)
        mask = mask[:, :, np.newaxis]
    else:
        mask = np.ones_like(hls)
    
    h += (hue / 2.0) * mask[:, :, 0]
    h = np.mod(h, 180.0)
    
    s_factor = 1.0 + saturation / 100.0
    s *= s_factor * (1.0 + (mask[:, :, 0] - 1.0) * 0.0)
    s = np.clip(s, 0, 255)
    
    l_factor = 1.0 + lightness / 100.0
    l_offset = (lightness / 100.0) * 128.0
    l += l_offset * mask[:, :, 0]
    l = np.clip(l, 0, 255)
    
    hls[:, :, 0] = h
    hls[:, :, 1] = l
    hls[:, :, 2] = s
    hls = np.clip(hls, 0, 255).astype(np.uint8)
    
    result = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    return cv_to_pil(result)


def adjust_color_balance(
    image: Image.Image,
    shadows: Tuple[int, int, int] = (0, 0, 0),
    midtones: Tuple[int, int, int] = (0, 0, 0),
    highlights: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """色彩平衡调整（阴影/中间调/高光）
    
    Args:
        shadows: (青-红, 洋红-绿, 黄-蓝) 各通道 -100 到 100
        midtones: 同上
        highlights: 同上
    """
    cv_img = pil_to_cv(image).astype(np.float32)
    
    lum = cv2.cvtColor(cv_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    shadow_mask = 1.0 - np.clip(lum / 0.33, 0, 1)
    highlight_mask = np.clip((lum - 0.67) / 0.33, 0, 1)
    midtone_mask = 1.0 - shadow_mask - highlight_mask
    
    masks = [shadow_mask, midtone_mask, highlight_mask]
    adjustments = [shadows, midtones, highlights]
    
    result = cv_img.copy()
    
    for mask, (cyan_red, magenta_green, yellow_blue) in zip(masks, adjustments):
        mask_3d = mask[:, :, np.newaxis]
        
        r_adj = cyan_red * 2.55
        g_adj = magenta_green * 2.55
        b_adj = yellow_blue * 2.55
        
        result[:, :, 2] += r_adj * mask
        result[:, :, 1] += g_adj * mask
        result[:, :, 0] += b_adj * mask
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv_to_pil(result)


def apply_vibrance(image: Image.Image, vibrance: float = 0.0) -> Image.Image:
    """自然饱和度调整（智能饱和度，保护已饱和颜色）"""
    cv_img = pil_to_cv(image).astype(np.float32)
    hsv = cv2.cvtColor(cv_img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    
    s = hsv[:, :, 1]
    s_normalized = s / 255.0
    
    vibrance_factor = vibrance / 100.0
    adjustment = (1.0 - s_normalized) * vibrance_factor * 255.0
    
    s_new = s + adjustment
    s_new = np.clip(s_new, 0, 255)
    
    hsv[:, :, 1] = s_new
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv_to_pil(result)


_LUT_CACHE = {}

_PRESET_LUTS = {
    "vintage": {
        "description": "复古胶片感",
        "rgb_points": [(0, 20), (64, 70), (128, 135), (192, 200), (255, 245)],
        "r_points": [(0, 0), (128, 138), (255, 250)],
        "b_points": [(0, 10), (128, 115), (255, 235)],
    },
    "warm_sunset": {
        "description": "暖色日落",
        "r_points": [(0, 0), (64, 80), (128, 150), (192, 215), (255, 255)],
        "b_points": [(0, 0), (128, 100), (255, 220)],
        "temperature": 25,
    },
    "cool_ocean": {
        "description": "冷调海洋",
        "b_points": [(0, 0), (64, 85), (128, 150), (192, 215), (255, 255)],
        "r_points": [(0, 0), (128, 110), (255, 240)],
        "temperature": -20,
    },
    "high_contrast": {
        "description": "高对比电影感",
        "rgb_points": [(0, 0), (48, 30), (128, 128), (207, 225), (255, 255)],
    },
    "soft_cream": {
        "description": "柔焦奶油",
        "rgb_points": [(0, 15), (64, 80), (128, 140), (192, 205), (255, 255)],
        "saturation": -10,
    },
    "bw_dramatic": {
        "description": "戏剧性黑白",
        "rgb_points": [(0, 0), (40, 15), (100, 90), (160, 180), (220, 245), (255, 255)],
        "grayscale": True,
    },
    "bw_classic": {
        "description": "经典黑白",
        "rgb_points": [(0, 0), (128, 128), (255, 255)],
        "grayscale": True,
    },
    "film_kodak": {
        "description": "柯达胶片",
        "r_points": [(0, 0), (64, 75), (128, 142), (192, 210), (255, 255)],
        "g_points": [(0, 5), (64, 72), (128, 135), (192, 200), (255, 245)],
        "b_points": [(0, 10), (64, 65), (128, 125), (192, 188), (255, 235)],
        "saturation": 8,
    },
    "matte_fade": {
        "description": "哑光褪色",
        "rgb_points": [(0, 30), (64, 85), (128, 140), (192, 200), (255, 240)],
        "contrast": -15,
        "saturation": -8,
    },
    "cinematic_teal": {
        "description": "电影青橙",
        "r_points": [(0, 0), (128, 120), (255, 255)],
        "b_points": [(0, 0), (128, 145), (255, 255)],
        "contrast": 10,
    },
}


def apply_preset_lut(image: Image.Image, preset_name: str) -> Image.Image:
    """应用预设LUT滤镜"""
    preset_name = preset_name.lower().replace(" ", "_")
    preset = _PRESET_LUTS.get(preset_name)
    if preset is None:
        return image
    
    result = image
    
    if preset.get("grayscale", False):
        result = result.convert("L").convert("RGB")
    
    if "rgb_points" in preset:
        result = adjust_curves(result, "rgb", preset["rgb_points"])
    if "r_points" in preset:
        result = adjust_curves(result, "r", preset["r_points"])
    if "g_points" in preset:
        result = adjust_curves(result, "g", preset["g_points"])
    if "b_points" in preset:
        result = adjust_curves(result, "b", preset["b_points"])
    
    if "contrast" in preset:
        result = adjust_curves(result, "rgb", None, 0, preset["contrast"])
    
    if "temperature" in preset or "tint" in preset:
        result = adjust_temperature_tint(
            result,
            temperature=preset.get("temperature", 0),
            tint=preset.get("tint", 0),
        )
    
    if "saturation" in preset:
        result = apply_vibrance(result, preset["saturation"])
    
    return result


def list_available_luts() -> Dict[str, str]:
    """列出所有可用的预设滤镜"""
    return {k: v["description"] for k, v in _PRESET_LUTS.items()}


def apply_lut_from_file(image: Image.Image, lut_file_path: str) -> Image.Image:
    """从CUBE格式LUT文件应用滤镜"""
    global _LUT_CACHE
    
    if lut_file_path in _LUT_CACHE:
        lut_data = _LUT_CACHE[lut_file_path]
    else:
        lut_data = _parse_cube_lut(lut_file_path)
        _LUT_CACHE[lut_file_path] = lut_data
    
    if lut_data is None:
        return image
    
    cv_img = pil_to_cv(image).astype(np.float32) / 255.0
    h, w = cv_img.shape[:2]
    
    lut_size = lut_data["size"]
    lut = lut_data["data"]
    
    b = np.clip(cv_img[:, :, 0] * (lut_size - 1), 0, lut_size - 1)
    g = np.clip(cv_img[:, :, 1] * (lut_size - 1), 0, lut_size - 1)
    r = np.clip(cv_img[:, :, 2] * (lut_size - 1), 0, lut_size - 1)
    
    b0 = np.floor(b).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    r0 = np.floor(r).astype(np.int32)
    b1 = np.minimum(b0 + 1, lut_size - 1)
    g1 = np.minimum(g0 + 1, lut_size - 1)
    r1 = np.minimum(r0 + 1, lut_size - 1)
    
    fb = b - b0.astype(np.float32)
    fg = g - g0.astype(np.float32)
    fr = r - r0.astype(np.float32)
    
    fb = fb[:, :, np.newaxis]
    fg = fg[:, :, np.newaxis]
    fr = fr[:, :, np.newaxis]
    
    idx_shape = r0.shape
    flat_idx = (r0, g0, b0)
    
    c000 = lut[r0, g0, b0]
    c100 = lut[r1, g0, b0]
    c010 = lut[r0, g1, b0]
    c001 = lut[r0, g0, b1]
    c110 = lut[r1, g1, b0]
    c101 = lut[r1, g0, b1]
    c011 = lut[r0, g1, b1]
    c111 = lut[r1, g1, b1]
    
    c00 = c000 * (1 - fr) + c100 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    
    result = c0 * (1 - fb) + c1 * fb
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return cv_to_pil(result)


def _parse_cube_lut(file_path: str) -> Optional[Dict[str, Any]]:
    """解析CUBE格式LUT文件"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None
    
    size = None
    title = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    values = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("TITLE"):
            title = line.split('"')[1] if '"' in line else line[5:].strip()
        elif line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
        elif line.upper().startswith("DOMAIN_MIN"):
            parts = line.split()
            domain_min = [float(x) for x in parts[1:4]]
        elif line.upper().startswith("DOMAIN_MAX"):
            parts = line.split()
            domain_max = [float(x) for x in parts[1:4]]
        else:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    values.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    
    if size is None or not values:
        return None
    
    expected_size = size ** 3
    if len(values) < expected_size:
        return None
    
    values = values[:expected_size]
    lut_array = np.array(values).reshape(size, size, size, 3)
    
    return {
        "size": size,
        "data": lut_array,
        "title": title,
        "domain_min": domain_min,
        "domain_max": domain_max,
    }


def adjust_basic(
    image: Image.Image,
    exposure: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
    saturation: float = 0.0,
    vibrance: float = 0.0,
) -> Image.Image:
    """综合基础调色（Lightroom风格）"""
    cv_img = pil_to_cv(image).astype(np.float32) / 255.0
    
    if exposure != 0:
        cv_img *= (2.0 ** (exposure / 100.0))
    
    if blacks != 0:
        blacks_val = blacks / 100.0
        cv_img = np.where(cv_img < 0.05, cv_img + blacks_val * 0.3, cv_img)
    
    if whites != 0:
        whites_val = whites / 100.0
        cv_img = np.where(cv_img > 0.95, cv_img + whites_val * 0.3, cv_img)
    
    lum = cv2.cvtColor((cv_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    if highlights != 0:
        hl_factor = highlights / 100.0
        hl_mask = np.clip((lum - 0.5) * 2.0, 0, 1)[:, :, np.newaxis]
        cv_img -= cv_img * hl_mask * hl_factor * 0.5
    
    if shadows != 0:
        sh_factor = shadows / 100.0
        sh_mask = np.clip((0.5 - lum) * 2.0, 0, 1)[:, :, np.newaxis]
        cv_img += cv_img * sh_mask * sh_factor * 0.5
    
    cv_img = np.clip(cv_img, 0, 1)
    
    result_uint8 = (cv_img * 255).astype(np.uint8)
    result = cv_to_pil(result_uint8)
    
    if brightness != 0 or contrast != 0:
        result = adjust_curves(result, "rgb", None, brightness, contrast)
    
    if saturation != 0:
        result = adjust_hsl(result, saturation=saturation)
    if vibrance != 0:
        result = apply_vibrance(result, vibrance)
    
    return result
