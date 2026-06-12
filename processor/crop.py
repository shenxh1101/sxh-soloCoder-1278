"""智能裁剪模块：人脸识别居中、主体检测、三分法构图、固定比例裁剪"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .core import pil_to_cv, cv_to_pil

_face_cascade = None
_face_detector_mediapipe = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        try:
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            _face_cascade = None
    return _face_cascade


def _get_mediapipe_face_detector():
    global _face_detector_mediapipe
    if _face_detector_mediapipe is None:
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            _face_detector_mediapipe = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except Exception:
            _face_detector_mediapipe = None
    return _face_detector_mediapipe


def detect_faces(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """检测图像中的人脸，返回人脸边界框列表 (x1, y1, x2, y2)"""
    cv_img = pil_to_cv(image)
    h, w = cv_img.shape[:2]
    
    faces = []
    
    mp_detector = _get_mediapipe_face_detector()
    if mp_detector is not None:
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = mp_detector.process(rgb_img)
            if results.detections:
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    faces.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
                return faces
        except Exception:
            pass
    
    cascade = _get_face_cascade()
    if cascade is not None:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        detected = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, fw, fh) in detected:
            faces.append((x, y, x + fw, y + fh))
    
    return faces


def detect_subject(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """基于显著性检测定位主体区域，返回边界框 (x1, y1, x2, y2)"""
    cv_img = pil_to_cv(image)
    h, w = cv_img.shape[:2]
    
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliency_map) = saliency.computeSaliency(cv_img)
        if not success:
            raise ValueError("Saliency computation failed")
    except Exception:
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliency_map) = saliency.computeSaliency(cv_img)
            if not success:
                return None
        except Exception:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            saliency_map = gray / 255.0
    
    saliency_uint8 = (saliency_map * 255).astype("uint8")
    _, thresh = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(max_contour)
    
    if cw * ch < (w * h) * 0.01:
        return None
    
    margin = int(min(cw, ch) * 0.15)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + cw + margin)
    y2 = min(h, y + ch + margin)
    
    return (x1, y1, x2, y2)


def _expand_crop_to_ratio(
    center_x: int, center_y: int, base_w: int, base_h: int,
    img_w: int, img_h: int, ratio: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    """以中心点为基准，按照指定比例(宽/高)扩展裁剪区域"""
    if ratio is None:
        ratio = base_w / base_h if base_h > 0 else 1.0
    
    half_w = base_w // 2
    half_h = base_h // 2
    
    if base_w / max(base_h, 1) > ratio:
        new_h = int(base_w / ratio)
        half_h = new_h // 2
    else:
        new_w = int(base_h * ratio)
        half_w = new_w // 2
    
    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h
    
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    cw = x2 - x1
    ch = y2 - y1
    target_h = int(cw / ratio) if ratio > 0 else ch
    if target_h <= ch and target_h > 0:
        diff = ch - target_h
        y1 += diff // 2
        y2 -= diff - diff // 2
    else:
        target_w = int(ch * ratio)
        if target_w <= cw and target_w > 0:
            diff = cw - target_w
            x1 += diff // 2
            x2 -= diff - diff // 2
    
    return (x1, y1, x2, y2)


def crop_face_centered(
    image: Image.Image,
    ratio: Optional[float] = None,
    padding_ratio: float = 2.5,
) -> Image.Image:
    """人脸居中裁剪：检测人脸并以人脸为中心进行裁剪"""
    w, h = image.size
    faces = detect_faces(image)
    
    if not faces:
        return crop_rule_of_thirds(image, ratio)
    
    if len(faces) == 1:
        fx1, fy1, fx2, fy2 = faces[0]
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        base_size = int(max(face_w, face_h) * padding_ratio)
    else:
        xs = [(f[0] + f[2]) // 2 for f in faces]
        ys = [(f[1] + f[3]) // 2 for f in faces]
        cx = sum(xs) // len(xs)
        cy = sum(ys) // len(ys)
        fx_min = min(f[0] for f in faces)
        fx_max = max(f[2] for f in faces)
        fy_min = min(f[1] for f in faces)
        fy_max = max(f[3] for f in faces)
        face_w = fx_max - fx_min
        face_h = fy_max - fy_min
        base_size = int(max(face_w, face_h) * padding_ratio * 1.5)
    
    base_w = base_size
    base_h = base_size
    if ratio is not None:
        if ratio >= 1:
            base_w = base_size
            base_h = int(base_size / ratio)
        else:
            base_h = base_size
            base_w = int(base_size * ratio)
    
    x1, y1, x2, y2 = _expand_crop_to_ratio(cx, cy, base_w, base_h, w, h, ratio)
    
    if (x2 - x1) < (w * 0.3) or (y2 - y1) < (h * 0.3):
        return crop_rule_of_thirds(image, ratio)
    
    return image.crop((x1, y1, x2, y2))


def crop_subject_centered(
    image: Image.Image,
    ratio: Optional[float] = None,
    padding_ratio: float = 1.5,
) -> Image.Image:
    """主体居中裁剪：基于显著性检测定位主体并居中裁剪"""
    w, h = image.size
    subject = detect_subject(image)
    
    if subject is None:
        return crop_rule_of_thirds(image, ratio)
    
    sx1, sy1, sx2, sy2 = subject
    cx = (sx1 + sx2) // 2
    cy = (sy1 + sy2) // 2
    subj_w = sx2 - sx1
    subj_h = sy2 - sy1
    base_w = int(subj_w * padding_ratio)
    base_h = int(subj_h * padding_ratio)
    
    x1, y1, x2, y2 = _expand_crop_to_ratio(cx, cy, base_w, base_h, w, h, ratio)
    
    if (x2 - x1) < (w * 0.3) or (y2 - y1) < (h * 0.3):
        return crop_rule_of_thirds(image, ratio)
    
    return image.crop((x1, y1, x2, y2))


def crop_rule_of_thirds(
    image: Image.Image,
    ratio: Optional[float] = None,
) -> Image.Image:
    """三分法构图裁剪：使用熵/边缘检测寻找最佳裁剪区域"""
    w, h = image.size
    
    if ratio is None:
        ratio = w / h
    
    target_w = w
    target_h = int(w / ratio)
    if target_h > h:
        target_h = h
        target_w = int(h * ratio)
    
    if target_w >= w and target_h >= h:
        target_w = w
        target_h = h
    
    cv_img = pil_to_cv(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_map = np.abs(edges)
    
    best_score = -1
    best_x1 = 0
    best_y1 = 0
    
    step = max(1, min(w - target_w, h - target_h) // 20)
    
    for y in range(0, h - target_h + 1, step):
        for x in range(0, w - target_w + 1, step):
            crop_edges = edge_map[y:y + target_h, x:x + target_w]
            score = np.mean(crop_edges)
            
            cx = x + target_w // 2
            cy = y + target_h // 2
            
            thirds_x = [x + target_w // 3, x + 2 * target_w // 3]
            thirds_y = [y + target_h // 3, y + 2 * target_h // 3]
            
            center_region = edge_map[
                max(0, cy - target_h // 6):min(h, cy + target_h // 6),
                max(0, cx - target_w // 6):min(w, cx + target_w // 6)
            ]
            if center_region.size > 0:
                center_score = np.mean(center_region)
                score -= center_score * 0.3
            
            for tx in thirds_x:
                for ty in thirds_y:
                    region = edge_map[
                        max(0, ty - 20):min(h, ty + 20),
                        max(0, tx - 20):min(w, tx + 20)
                    ]
                    if region.size > 0:
                        score += np.mean(region) * 0.5
            
            if score > best_score:
                best_score = score
                best_x1 = x
                best_y1 = y
    
    return image.crop((best_x1, best_y1, best_x1 + target_w, best_y1 + target_h))


def crop_fixed_ratio(
    image: Image.Image,
    ratio: float,
    mode: str = "center",
) -> Image.Image:
    """固定比例裁剪
    
    Args:
        image: 输入图像
        ratio: 宽/高比例，例如 1.0 (正方形), 1.33 (4:3), 1.78 (16:9), 0.75 (3:4)
        mode: 裁剪模式 - center(中心), top(顶部), bottom(底部), left(左侧), right(右侧)
    """
    w, h = image.size
    current_ratio = w / h
    
    if abs(current_ratio - ratio) < 0.001:
        return image
    
    target_w = w
    target_h = int(w / ratio)
    
    if target_h > h:
        target_h = h
        target_w = int(h * ratio)
    
    x1 = y1 = 0
    
    if mode == "center":
        x1 = (w - target_w) // 2
        y1 = (h - target_h) // 2
    elif mode == "top":
        x1 = (w - target_w) // 2
        y1 = 0
    elif mode == "bottom":
        x1 = (w - target_w) // 2
        y1 = h - target_h
    elif mode == "left":
        x1 = 0
        y1 = (h - target_h) // 2
    elif mode == "right":
        x1 = w - target_w
        y1 = (h - target_h) // 2
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    return image.crop((x1, y1, x1 + target_w, y1 + target_h))


def smart_crop(
    image: Image.Image,
    mode: str = "rule_of_thirds",
    ratio: Optional[float] = None,
    **kwargs,
) -> Image.Image:
    """统一的智能裁剪入口
    
    Args:
        image: 输入图像
        mode: 裁剪模式 - face, subject, rule_of_thirds, fixed
        ratio: 宽高比 (可选)
        **kwargs: 各模式特有参数
    """
    mode = mode.lower()
    
    if mode == "face":
        return crop_face_centered(image, ratio=ratio, **kwargs)
    elif mode == "subject":
        return crop_subject_centered(image, ratio=ratio, **kwargs)
    elif mode == "rule_of_thirds":
        return crop_rule_of_thirds(image, ratio=ratio)
    elif mode == "fixed":
        return crop_fixed_ratio(image, ratio=ratio if ratio else 1.0, **kwargs)
    else:
        return image
