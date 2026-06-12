"""AI辅助功能模块：自动去背景、智能锐化、老照片修复、风格迁移"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image, ImageFilter
import cv2
from scipy import ndimage

from .core import pil_to_cv, cv_to_pil


_REMBG_SESSION = None


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            _REMBG_SESSION = new_session()
        except Exception:
            _REMBG_SESSION = False
    return _REMBG_SESSION


def remove_background_rembg(image: Image.Image, alpha_matting: bool = True) -> Image.Image:
    """使用rembg库去背景（深度学习）"""
    session = _get_rembg_session()
    if not session:
        return remove_background_saliency(image)
    
    try:
        from rembg import remove
        result = remove(image, session=session, alpha_matting=alpha_matting)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result
    except Exception:
        return remove_background_saliency(image)


def remove_background_saliency(image: Image.Image) -> Image.Image:
    """基于显著性检测 + GrabCut 的传统方法去背景"""
    cv_img = pil_to_cv(image)
    h, w = cv_img.shape[:2]
    
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliency_map) = saliency.computeSaliency(cv_img)
        if not success:
            raise ValueError("Saliency failed")
    except Exception:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        saliency_map = edges.astype(np.float32) / 255.0
    
    sal_uint8 = (saliency_map * 255).astype(np.uint8)
    _, fg_mask = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if np.sum(fg_mask > 0) < (h * w * 0.05):
        _, fg_mask = cv2.threshold(sal_uint8, np.percentile(sal_uint8, 70), 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    probable_fg = cv2.dilate(fg_mask, kernel_dil, iterations=3)
    unknown = cv2.subtract(probable_fg, sure_fg)
    
    markers = np.zeros((h, w), np.int32)
    markers[fg_mask == 0] = cv2.GC_BGD
    markers[unknown > 0] = cv2.GC_PR_BGD
    markers[sure_fg > 0] = cv2.GC_FGD
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(cv_img, markers, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        pass
    
    mask = np.where((markers == cv2.GC_FGD) | (markers == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_float = mask.astype(np.float32) / 255.0
    
    if image.mode == "RGBA":
        rgb = np.array(image)[:, :, :3]
    else:
        rgb = np.array(image.convert("RGB"))
    
    alpha = (mask_float * 255).astype(np.uint8)
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))
    
    result = np.dstack([rgb, alpha])
    return Image.fromarray(result, "RGBA")


def remove_background(
    image: Image.Image,
    method: str = "auto",
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """自动去背景
    
    Args:
        image: 输入图像
        method: auto, ai(使用rembg), traditional(显著性+GrabCut)
        bg_color: 如果指定，将背景替换为此颜色而不是透明
    """
    method = method.lower()
    
    if method == "auto":
        result = remove_background_rembg(image)
        if result is None or (result.mode == "RGBA" and np.mean(np.array(result)[:, :, 3]) > 250):
            result = remove_background_saliency(image)
    elif method == "ai":
        result = remove_background_rembg(image)
    else:
        result = remove_background_saliency(image)
    
    if bg_color is not None:
        bg = Image.new("RGB", result.size, bg_color)
        if result.mode == "RGBA":
            bg.paste(result, mask=result.split()[-1])
        else:
            bg.paste(result)
        return bg
    
    return result


def _estimate_blur_level(cv_img: np.ndarray) -> float:
    """估计图像模糊程度（拉普拉斯方差），值越低越模糊"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def deblur_wiener(cv_img: np.ndarray, kernel_size: int = 5, snr: float = 0.01) -> np.ndarray:
    """维纳滤波去模糊"""
    result_channels = []
    for c in range(cv_img.shape[2]):
        img = cv_img[:, :, c].astype(np.float64) / 255.0
        
        psf = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size * kernel_size)
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        
        psf_fft = np.fft.fft2(psf_pad)
        img_fft = np.fft.fft2(img)
        
        wiener = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 1.0 / snr)
        deblur_fft = img_fft * wiener
        deblur = np.fft.ifft2(deblur_fft)
        deblur = np.abs(np.fft.fftshift(deblur))
        
        result_channels.append(np.clip(deblur, 0, 1))
    
    return (np.dstack(result_channels) * 255).astype(np.uint8)


def deblur_richardson_lucy(cv_img: np.ndarray, iterations: int = 30, psf_size: int = 5) -> np.ndarray:
    """Richardson-Lucy去模糊算法"""
    psf = np.ones((psf_size, psf_size), np.float64) / (psf_size * psf_size)
    psf_mirror = psf[::-1, ::-1]
    
    result_channels = []
    for c in range(cv_img.shape[2]):
        img = cv_img[:, :, c].astype(np.float64) / 255.0
        estimate = np.full_like(img, 0.5)
        
        for _ in range(iterations):
            est_conv = ndimage.convolve(estimate, psf, mode="mirror")
            relative_blur = img / (est_conv + 1e-10)
            error_est = ndimage.convolve(relative_blur, psf_mirror, mode="mirror")
            estimate *= error_est
        
        result_channels.append(np.clip(estimate, 0, 1))
    
    return (np.dstack(result_channels) * 255).astype(np.uint8)


def smart_sharpen(
    image: Image.Image,
    strength: Optional[float] = None,
    radius: float = 1.0,
) -> Image.Image:
    """智能锐化（自动检测模糊程度并去模糊+USM锐化）
    
    Args:
        image: 输入图像
        strength: 锐化强度 (0.0-2.0)，None为自动
        radius: 锐化半径
    """
    cv_img = pil_to_cv(image)
    blur_level = _estimate_blur_level(cv_img)
    
    if strength is None:
        if blur_level < 50:
            strength = 1.8
        elif blur_level < 150:
            strength = 1.4
        elif blur_level < 300:
            strength = 1.0
        else:
            strength = 0.6
    
    deblurred = cv_img
    if blur_level < 300:
        try:
            ks = 5 if blur_level < 100 else 3
            iters = 20 if blur_level < 100 else 10
            deblurred = deblur_richardson_lucy(cv_img, iterations=iters, psf_size=ks)
        except Exception:
            deblurred = cv_img
    
    result = deblurred.copy()
    for c in range(result.shape[2]):
        channel = deblurred[:, :, c]
        blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=radius)
        sharpened = cv2.addWeighted(channel, 1 + strength, blurred, -strength, 0)
        result[:, :, c] = sharpened
    
    amount = max(0, strength - 0.5) * 0.5
    if amount > 0:
        for c in range(result.shape[2]):
            channel = result[:, :, c]
            blur1 = cv2.GaussianBlur(channel, (0, 0), sigmaX=0.5)
            blur2 = cv2.GaussianBlur(channel, (0, 0), sigmaX=radius * 2)
            result[:, :, c] = cv2.addWeighted(channel, 1 + amount, blur2, -amount, 0)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv_to_pil(result)


def unsharp_mask(
    image: Image.Image,
    amount: float = 1.0,
    radius: float = 1.5,
    threshold: int = 2,
) -> Image.Image:
    """标准USM锐化"""
    pil_radius = max(0.1, radius)
    return image.filter(ImageFilter.UnsharpMask(radius=pil_radius, percent=int(amount * 100), threshold=threshold))


def detect_scratches(cv_img: np.ndarray) -> np.ndarray:
    """检测老照片中的划痕"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(np.abs(grad_y), np.abs(grad_x)) * 180.0 / np.pi
    
    _, mag_binary = cv2.threshold(magnitude, np.percentile(magnitude, 97), 255, cv2.THRESH_BINARY)
    mag_binary = mag_binary.astype(np.uint8)
    
    vertical_mask = (direction > 60).astype(np.uint8) * 255
    horizontal_mask = (direction < 30).astype(np.uint8) * 255
    
    vertical_lines = cv2.bitwise_and(mag_binary, vertical_mask)
    horizontal_lines = cv2.bitwise_and(mag_binary, horizontal_mask)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel_v, iterations=2)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel_h, iterations=2)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    
    scratch_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_mask = cv2.dilate(scratch_mask, kernel_dilate, iterations=1)
    
    contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(scratch_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if aspect > 3:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered_mask = cv2.dilate(filtered_mask, kernel_final, iterations=1)
    
    return filtered_mask


def restore_old_photo(
    image: Image.Image,
    remove_scratches: bool = True,
    denoise: bool = True,
    color_correct: bool = True,
    enhance_contrast: bool = True,
) -> Image.Image:
    """老照片修复：划痕填补 + 去噪 + 色彩校正 + 对比度增强
    
    Args:
        image: 输入图像（可能是老照片）
        remove_scratches: 是否去除划痕
        denoise: 是否降噪
        color_correct: 是否校正褪色
        enhance_contrast: 是否增强对比度
    """
    cv_img = pil_to_cv(image)
    h, w = cv_img.shape[:2]
    
    if remove_scratches:
        scratch_mask = detect_scratches(cv_img)
        
        if np.sum(scratch_mask > 0) > 0:
            if np.sum(scratch_mask > 0) < (h * w * 0.1):
                repaired = cv2.inpaint(cv_img, scratch_mask, 3, cv2.INPAINT_TELEA)
            else:
                repaired = cv2.inpaint(cv_img, scratch_mask, 5, cv2.INPAINT_NS)
            cv_img = repaired
    
    if denoise:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        noise_level = np.std(gray[:h//4, :w//4])
        
        if noise_level > 15 or np.mean(gray) < 100:
            cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
    
    if color_correct:
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        if abs(a_mean - 128) > 10 or abs(b_mean - 128) > 10:
            a = cv2.add(a, int(128 - a_mean) // 2)
            b = cv2.add(b, int(128 - b_mean) // 2)
        
        lab = cv2.merge([l, a, b])
        cv_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        from .color import _gray_world_wb
        cv_img = _gray_world_wb(cv_img)
    
    if enhance_contrast:
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        cv_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    result = cv_to_pil(cv_img)
    
    if image.mode == "L" or (image.mode == "RGB" and _is_sepia_or_grayscale(image)):
        result = _add_film_grain(result, amount=0.02)
    
    return result


def _is_sepia_or_grayscale(image: Image.Image) -> bool:
    """检测是否为灰度或棕褐色调图像"""
    arr = np.array(image)
    if len(arr.shape) < 3:
        return True
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    diff_rg = np.mean(np.abs(r.astype(int) - g.astype(int)))
    diff_gb = np.mean(np.abs(g.astype(int) - b.astype(int)))
    return diff_rg < 20 and diff_gb < 20


def _add_film_grain(image: Image.Image, amount: float = 0.02) -> Image.Image:
    """添加细腻胶片颗粒感"""
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 255 * amount, arr.shape)
    result = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def style_oil_painting(image: Image.Image, strength: int = 5) -> Image.Image:
    """油画风格转换"""
    cv_img = pil_to_cv(image)
    strength = max(1, min(strength, 10))
    result = cv2.xphoto.oilPainting(cv_img, size=strength, dynRatio=1)
    return cv_to_pil(result)


def style_watercolor(image: Image.Image, strength: float = 0.7) -> Image.Image:
    """水彩风格转换"""
    cv_img = pil_to_cv(image)
    
    bilateral = cv2.bilateralFilter(cv_img, d=9, sigmaColor=150, sigmaSpace=150)
    
    detail_enhanced = cv2.detailEnhance(bilateral, sigma_s=10, sigma_r=0.15)
    
    lab = cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    detail_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    edge_preserve = cv2.edgePreservingFilter(detail_enhanced, flags=1, sigma_s=60, sigma_r=0.4)
    
    result = cv2.addWeighted(cv_img, 1 - strength, edge_preserve, strength, 0)
    
    return cv_to_pil(result)


def style_sketch_pencil(image: Image.Image, color: bool = False) -> Image.Image:
    """素描风格转换
    
    Args:
        image: 输入图像
        color: True=彩色铅笔素描，False=黑白铅笔素描
    """
    cv_img = pil_to_cv(image)
    
    if color:
        sketch_bgr, sketch_color = cv2.pencilSketch(
            cv_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
        )
        result = sketch_color
    else:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
        result = cv2.divide(gray, 255 - blur, scale=256)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return cv_to_pil(result)


def style_cinematic(image: Image.Image) -> Image.Image:
    """电影风格：暗角 + 高对比 + 青橙色调"""
    cv_img = pil_to_cv(image).astype(np.float32) / 255.0
    h, w = cv_img.shape[:2]
    
    cv_img[:, :, 2] = np.power(cv_img[:, :, 2], 1.05)
    cv_img[:, :, 0] = np.power(cv_img[:, :, 0], 0.95)
    
    lab = cv2.cvtColor((cv_img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    l_mean = np.mean(l)
    l = np.where(l > l_mean, l + (l - l_mean) * 0.2, l - (l_mean - l) * 0.15)
    lab[:, :, 0] = np.clip(l, 0, 255)
    cv_img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    
    Y = np.linspace(-1, 1, h)[:, np.newaxis]
    X = np.linspace(-1, 1, w)[np.newaxis, :]
    radius = np.sqrt(X ** 2 + Y ** 2)
    vignette = 1 - np.clip(radius - 0.5, 0, 1) ** 2 * 0.7
    vignette = vignette[:, :, np.newaxis]
    
    cv_img = cv_img * vignette
    result = np.clip(cv_img * 255, 0, 255).astype(np.uint8)
    return cv_to_pil(result)


def style_comic_book(image: Image.Image) -> Image.Image:
    """漫画/卡通风格"""
    cv_img = pil_to_cv(image)
    
    color = cv2.bilateralFilter(cv_img, d=9, sigmaColor=300, sigmaSpace=300)
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    result = cv2.bitwise_and(color, edges)
    return cv_to_pil(result)


def style_vintage_film(image: Image.Image) -> Image.Image:
    """复古胶片风格"""
    from .color import apply_preset_lut
    result = apply_preset_lut(image, "vintage")
    cv_result = pil_to_cv(result).astype(np.float32) / 255.0
    h, w = cv_result.shape[:2]
    
    Y = np.linspace(-1, 1, h)[:, np.newaxis]
    X = np.linspace(-1, 1, w)[np.newaxis, :]
    radius = np.sqrt(X ** 2 + Y ** 2)
    vignette = 1 - np.clip(radius - 0.4, 0, 1) ** 2 * 0.6
    vignette = vignette[:, :, np.newaxis]
    
    cv_result = cv_result * vignette
    
    noise = np.random.normal(0, 0.015, cv_result.shape)
    cv_result = np.clip(cv_result + noise, 0, 1)
    
    scratches = np.random.choice([0, 1], size=cv_result.shape[:2], p=[0.999, 0.001]).astype(np.float32)
    scratches = cv2.dilate(scratches, np.ones((1, 20), np.uint8), iterations=1)
    scratches = scratches[:, :, np.newaxis]
    cv_result = np.where(scratches > 0, 1.0, cv_result)
    
    result = (cv_result * 255).astype(np.uint8)
    return cv_to_pil(result)


def apply_style_transfer(image: Image.Image, style: str, **kwargs) -> Image.Image:
    """统一风格迁移入口
    
    Args:
        image: 输入图像
        style: 风格名称 - oil, watercolor, sketch, sketch_color, cinematic, comic, vintage
        **kwargs: 各风格参数
    """
    style = style.lower().replace(" ", "_")
    
    if style in ("oil", "oil_painting", "油画"):
        return style_oil_painting(image, kwargs.get("strength", 5))
    elif style in ("watercolor", "水彩"):
        return style_watercolor(image, kwargs.get("strength", 0.7))
    elif style in ("sketch", "pencil", "素描"):
        return style_sketch_pencil(image, color=False)
    elif style in ("sketch_color", "彩色素描", "color_pencil"):
        return style_sketch_pencil(image, color=True)
    elif style in ("cinematic", "电影"):
        return style_cinematic(image)
    elif style in ("comic", "cartoon", "漫画", "卡通"):
        return style_comic_book(image)
    elif style in ("vintage", "retro", "复古", "复古胶片"):
        return style_vintage_film(image)
    else:
        return image


def list_available_styles() -> Dict[str, str]:
    """列出所有可用的风格"""
    return {
        "oil": "油画效果",
        "watercolor": "水彩效果",
        "sketch": "黑白素描",
        "sketch_color": "彩色铅笔素描",
        "cinematic": "电影感（暗角+青橙）",
        "comic": "漫画/卡通",
        "vintage": "复古胶片",
    }
