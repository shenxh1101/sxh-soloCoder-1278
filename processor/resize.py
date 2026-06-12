"""尺寸处理模块：批量缩放、分辨率统一、长边/短边限制、智能填充背景色"""

from typing import Tuple, Optional, List, Union
import numpy as np
from PIL import Image
import cv2

from .core import pil_to_cv, cv_to_pil


def resize_by_scale(
    image: Image.Image,
    scale: float,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """按比例缩放图像
    
    Args:
        scale: 缩放比例，1.0=原图，0.5=缩小一半，2.0=放大一倍
        resample: 重采样方法
    """
    if abs(scale - 1.0) < 1e-6:
        return image
    
    w, h = image.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), resample)


def resize_to_width(
    image: Image.Image,
    target_width: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """按指定宽度等比缩放"""
    w, h = image.size
    if w == target_width:
        return image
    ratio = target_width / w
    new_h = max(1, int(h * ratio))
    return image.resize((target_width, new_h), resample)


def resize_to_height(
    image: Image.Image,
    target_height: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """按指定高度等比缩放"""
    w, h = image.size
    if h == target_height:
        return image
    ratio = target_height / h
    new_w = max(1, int(w * ratio))
    return image.resize((new_w, target_height), resample)


def resize_long_edge(
    image: Image.Image,
    max_long_edge: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """限制长边尺寸，等比缩放"""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    return resize_by_scale(image, scale, resample)


def resize_short_edge(
    image: Image.Image,
    min_short_edge: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """限制短边最小尺寸，等比缩放"""
    w, h = image.size
    short_edge = min(w, h)
    if short_edge >= min_short_edge:
        return image
    scale = min_short_edge / short_edge
    return resize_by_scale(image, scale, resample)


def resize_exact(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample: int = Image.LANCZOS,
    allow_upscale: bool = True,
) -> Image.Image:
    """强制缩放到精确尺寸（不保持比例）"""
    w, h = image.size
    if not allow_upscale and (target_size[0] > w or target_size[1] > h):
        target_w = min(w, target_size[0])
        target_h = min(h, target_size[1])
    else:
        target_w, target_h = target_size
    return image.resize((target_w, target_h), resample)


def resize_fit(
    image: Image.Image,
    max_size: Tuple[int, int],
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """按比例缩放以适应边界框（保持比例，不填充）"""
    w, h = image.size
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return image
    return resize_by_scale(image, scale, resample)


def _get_dominant_color(cv_img: np.ndarray) -> Tuple[int, int, int]:
    """获取图像的主色调"""
    small = cv2.resize(cv_img, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    
    from sklearn.cluster import MiniBatchKMeans
    try:
        kmeans = MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]
        return tuple(int(c) for c in dominant)
    except Exception:
        avg = np.mean(pixels, axis=0)
        return tuple(int(c) for c in avg)


def _get_border_color(cv_img: np.ndarray) -> Tuple[int, int, int]:
    """获取图像边缘的平均颜色"""
    h, w = cv_img.shape[:2]
    border_thickness = max(1, min(h, w) // 20)
    
    borders = [
        cv_img[:border_thickness, :, :].reshape(-1, 3),
        cv_img[-border_thickness:, :, :].reshape(-1, 3),
        cv_img[:, :border_thickness, :].reshape(-1, 3),
        cv_img[:, -border_thickness:, :].reshape(-1, 3),
    ]
    all_border = np.vstack(borders)
    avg = np.mean(all_border, axis=0)
    return tuple(int(c) for c in avg)


def smart_pad_to_size(
    image: Image.Image,
    target_size: Tuple[int, int],
    mode: str = "auto",
    bg_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    blur_strength: int = 50,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """智能填充背景到指定尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (宽, 高)
        mode: 填充模式
            - auto: 自动选择（优先边缘色，其次模糊延伸）
            - color: 纯色填充
            - blur: 模糊延伸填充
            - mirror: 镜像填充
            - repeat: 重复填充
            - dominant: 主色调填充
        bg_color: 纯色填充时的背景色，如 (255,255,255) 或 "white"
        blur_strength: 模糊填充时的模糊强度
    """
    w, h = image.size
    target_w, target_h = target_size
    
    if w == target_w and h == target_h:
        return image
    
    fitted_img = resize_fit(image, target_size, resample)
    fw, fh = fitted_img.size
    
    offset_x = (target_w - fw) // 2
    offset_y = (target_h - fh) // 2
    
    if image.mode != "RGB":
        fitted_img = fitted_img.convert("RGB")
    else:
        fitted_img = fitted_img.copy()
    
    if mode == "auto":
        cv_fitted = pil_to_cv(fitted_img)
        border_color = _get_border_color(cv_fitted)
        if _is_color_uniform(border_color):
            mode = "color"
            bg_color = border_color
        else:
            mode = "blur"
    
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    
    if mode == "color":
        if bg_color is None:
            bg_color = (255, 255, 255)
        elif isinstance(bg_color, str):
            from PIL import ImageColor
            bg_color = ImageColor.getrgb(bg_color)
        canvas = Image.new("RGB", (target_w, target_h), bg_color)
    
    elif mode == "dominant":
        cv_fitted = pil_to_cv(fitted_img)
        dominant = _get_dominant_color(cv_fitted)
        canvas = Image.new("RGB", (target_w, target_h), dominant)
    
    elif mode == "blur":
        bg_img = resize_exact(image, target_size, resample)
        cv_bg = pil_to_cv(bg_img)
        ksize = max(3, blur_strength // 5 * 2 + 1)
        cv_bg_blurred = cv2.GaussianBlur(cv_bg, (ksize, ksize), 0)
        canvas = cv_to_pil(cv_bg_blurred)
    
    elif mode == "mirror":
        bg_img = resize_exact(image, target_size, resample)
        cv_bg = pil_to_cv(bg_img)
        cv_bg = cv2.copyMakeBorder(cv_bg, 0, 0, 0, 0, cv2.BORDER_REFLECT_101)
        ksize = max(3, 51)
        cv_bg = cv2.GaussianBlur(cv_bg, (ksize, ksize), 0)
        canvas = cv_to_pil(cv_bg)
    
    elif mode == "repeat":
        bg_img = image.copy()
        cv_bg = pil_to_cv(bg_img)
        tile_h = target_h // cv_bg.shape[0] + 2
        tile_w = target_w // cv_bg.shape[1] + 2
        cv_tiled = np.tile(cv_bg, (tile_h, tile_w, 1))
        cv_tiled = cv_tiled[:target_h, :target_w]
        ksize = max(3, 41)
        cv_tiled = cv2.GaussianBlur(cv_tiled, (ksize, ksize), 0)
        canvas = cv_to_pil(cv_tiled)
    
    mask = Image.new("L", (target_w, target_h), 0)
    mask.paste(Image.new("L", (fw, fh), 255), (offset_x, offset_y))
    canvas.paste(fitted_img, (offset_x, offset_y))
    
    return canvas


def _is_color_uniform(color: Tuple[int, int, int], threshold: int = 20) -> bool:
    """判断颜色是否接近中性色（用于auto模式判断）"""
    r, g, b = color
    return max(r, g, b) - min(r, g, b) < threshold


def resize_uniform_resolution(
    image: Image.Image,
    target_mp: float = 24.0,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """统一分辨率到指定百万像素数
    
    Args:
        target_mp: 目标百万像素数，如24表示24MP
    """
    w, h = image.size
    current_mp = (w * h) / 1_000_000
    
    if abs(current_mp - target_mp) < 0.1:
        return image
    
    scale = (target_mp / current_mp) ** 0.5
    return resize_by_scale(image, scale, resample)


def add_border(
    image: Image.Image,
    border_width: Union[int, Tuple[int, int, int, int]] = 10,
    color: Union[str, Tuple[int, int, int]] = "white",
) -> Image.Image:
    """添加边框"""
    if isinstance(border_width, int):
        left = right = top = bottom = border_width
    else:
        left, top, right, bottom = border_width
    
    w, h = image.size
    new_w = w + left + right
    new_h = h + top + bottom
    
    if isinstance(color, str):
        from PIL import ImageColor
        color = ImageColor.getrgb(color)
    
    new_img = Image.new(image.mode, (new_w, new_h), color)
    new_img.paste(image, (left, top))
    return new_img


def add_watermark(
    image: Image.Image,
    watermark: Image.Image,
    position: str = "bottom-right",
    opacity: float = 0.5,
    margin: int = 20,
    scale: float = 0.2,
) -> Image.Image:
    """添加水印"""
    w, h = image.size
    wm_w = max(1, int(w * scale))
    wm_h = max(1, int(watermark.height * (wm_w / watermark.width)))
    wm = watermark.resize((wm_w, wm_h), Image.LANCZOS)
    
    if wm.mode != "RGBA":
        wm = wm.convert("RGBA")
    
    if opacity < 1.0:
        alpha = wm.split()[-1]
        alpha = alpha.point(lambda p: int(p * opacity))
        wm.putalpha(alpha)
    
    if position == "bottom-right":
        pos = (w - wm_w - margin, h - wm_h - margin)
    elif position == "bottom-left":
        pos = (margin, h - wm_h - margin)
    elif position == "top-right":
        pos = (w - wm_w - margin, margin)
    elif position == "top-left":
        pos = (margin, margin)
    elif position == "center":
        pos = ((w - wm_w) // 2, (h - wm_h) // 2)
    else:
        pos = (w - wm_w - margin, h - wm_h - margin)
    
    result = image.convert("RGBA")
    result.paste(wm, pos, wm)
    if image.mode != "RGBA":
        result = result.convert(image.mode)
    return result
