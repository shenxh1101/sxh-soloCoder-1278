"""核心图片处理工具：加载、保存、格式转换、EXIF处理等"""

import os
import io
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, ExifTags, TiffImagePlugin
from PIL.ExifTags import TAGS, GPSTAGS

try:
    import piexif
    _HAS_PIEXIF = True
except ImportError:
    _HAS_PIEXIF = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except ImportError:
    _HAS_HEIF = False

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif",
    ".heic", ".heif", ".avif", ".gif"
}


@dataclass
class ImageInfo:
    path: str
    filename: str
    width: int
    height: int
    format: str
    mode: str
    size_mb: float
    exif: Dict[str, Any] = field(default_factory=dict)


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """将PIL图像转换为OpenCV格式(BGR)"""
    if pil_image.mode == "RGBA":
        cv_image = np.array(pil_image)
        return cv_image[:, :, [2, 1, 0, 3]]
    elif pil_image.mode == "RGB":
        cv_image = np.array(pil_image)
        return cv_image[:, :, ::-1].copy()
    else:
        return np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()


def cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """将OpenCV图像转换为PIL格式"""
    if len(cv_image.shape) == 2:
        return Image.fromarray(cv_image)
    elif cv_image.shape[2] == 4:
        return Image.fromarray(cv_image[:, :, [2, 1, 0, 3]])
    else:
        return Image.fromarray(cv_image[:, :, ::-1].copy())


def load_image(filepath: Union[str, Path]) -> Image.Image:
    """加载图像文件，支持多种格式"""
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    
    if ext in (".heic", ".heif"):
        img = Image.open(filepath)
    else:
        img = Image.open(filepath)
    
    img = ImageOps.exif_transpose(img)
    return img


def _decode_exif_bytes(exif_bytes: bytes) -> Dict[str, Any]:
    """解码EXIF字节数据"""
    result = {}
    if not _HAS_PIEXIF:
        return result
    try:
        exif_dict = piexif.load(exif_bytes)
        for ifd in ("0th", "Exif", "GPS", "1st"):
            if ifd not in exif_dict:
                continue
            for tag_id, value in exif_dict[ifd].items():
                tag_name = piexif.TAGS.get(ifd, {}).get(tag_id, {}).get("name", str(tag_id))
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except:
                        value = str(value)
                result[f"{ifd}.{tag_name}"] = value
    except Exception:
        pass
    return result


def read_exif(image: Union[Image.Image, str, Path]) -> Dict[str, Any]:
    """读取EXIF信息"""
    if isinstance(image, (str, Path)):
        try:
            img = Image.open(str(image))
            exif_data = img.info.get("exif")
        except Exception:
            return {}
    else:
        exif_data = image.info.get("exif")
    
    if not exif_data:
        return {}
    return _decode_exif_bytes(exif_data)


def write_exif(image: Image.Image, exif_data: Dict[str, Any]) -> Image.Image:
    """写入EXIF信息到图像"""
    if not _HAS_PIEXIF:
        return image
    try:
        exif_bytes = image.info.get("exif")
        if exif_bytes:
            exif_dict = piexif.load(exif_bytes)
        else:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        for key, value in exif_data.items():
            parts = key.split(".", 1)
            if len(parts) != 2:
                continue
            ifd, tag_name = parts
            if ifd not in exif_dict:
                continue
            tag_id = None
            for tid, tinfo in piexif.TAGS.get(ifd, {}).items():
                if tinfo.get("name") == tag_name:
                    tag_id = tid
                    break
            if tag_id is not None:
                exif_dict[ifd][tag_id] = value
        
        new_exif_bytes = piexif.dump(exif_dict)
        image.info["exif"] = new_exif_bytes
    except Exception:
        pass
    return image


def strip_exif(image: Image.Image) -> Image.Image:
    """剥离EXIF信息"""
    data = list(image.getdata())
    new_img = Image.new(image.mode, image.size)
    new_img.putdata(data)
    return new_img


def save_image(
    image: Image.Image,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95,
    optimize: bool = True,
    keep_exif: bool = True,
) -> None:
    """保存图像，支持多种格式和参数"""
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    
    save_kwargs = {}
    
    if format is None:
        format = ext.lstrip(".").upper()
        if format == "JPG":
            format = "JPEG"
        elif format == "TIF":
            format = "TIFF"
    
    if not keep_exif and "exif" in image.info:
        save_kwargs["exif"] = b""
    elif "exif" in image.info and image.info["exif"]:
        save_kwargs["exif"] = image.info["exif"]
    
    if format in ("JPEG", "JPG"):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        save_kwargs.update({"quality": quality, "optimize": optimize, "progressive": True})
    
    elif format == "PNG":
        if optimize:
            save_kwargs.update({"optimize": True, "compress_level": 9})
    
    elif format == "WEBP":
        save_kwargs.update({"quality": quality, "method": 6})
    
    elif format in ("HEIC", "HEIF"):
        save_kwargs.update({"quality": quality})
    
    elif format in ("TIFF", "TIF"):
        save_kwargs.update({"compression": "tiff_lzw" if optimize else None})
    
    image.save(filepath, format=format, **save_kwargs)


def get_image_info(filepath: Union[str, Path]) -> ImageInfo:
    """获取图像文件信息"""
    filepath = str(filepath)
    path_obj = Path(filepath)
    img = Image.open(filepath)
    
    exif = read_exif(img)
    size_mb = path_obj.stat().st_size / (1024 * 1024)
    
    return ImageInfo(
        path=filepath,
        filename=path_obj.name,
        width=img.width,
        height=img.height,
        format=img.format or path_obj.suffix.lstrip(".").upper(),
        mode=img.mode,
        size_mb=round(size_mb, 2),
        exif=exif,
    )


def list_images_in_dir(directory: Union[str, Path], recursive: bool = True) -> List[str]:
    """列出目录中的所有图像文件"""
    directory = Path(directory)
    images = []
    
    pattern = "**/*" if recursive else "*"
    
    for f in directory.glob(pattern):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(str(f))
    
    return sorted(images)


def extract_zip(zip_path: Union[str, Path], extract_dir: Optional[Union[str, Path]] = None) -> str:
    """解压ZIP文件，返回解压目录路径"""
    zip_path = str(zip_path)
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="image_batch_")
    else:
        extract_dir = str(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    
    return extract_dir


def create_zip(
    source_dir: Union[str, Path],
    zip_path: Union[str, Path],
    progress_callback=None,
) -> str:
    """将目录打包为ZIP"""
    source_dir = Path(source_dir)
    zip_path = str(zip_path)
    
    all_files = [f for f in source_dir.rglob("*") if f.is_file()]
    total = len(all_files)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, f in enumerate(all_files, 1):
            arcname = f.relative_to(source_dir)
            zf.write(str(f), arcname=str(arcname))
            if progress_callback:
                progress_callback(idx / total)
    
    return zip_path


def generate_thumbnail(
    image: Union[Image.Image, str, Path],
    max_size: Tuple[int, int] = (300, 300),
) -> Image.Image:
    """生成缩略图"""
    if isinstance(image, (str, Path)):
        img = load_image(image)
    else:
        img = image.copy()
    
    img.thumbnail(max_size, Image.LANCZOS)
    return img


def image_to_bytes(image: Image.Image, format: str = "PNG", quality: int = 95) -> bytes:
    """将PIL图像转换为字节"""
    buf = io.BytesIO()
    save_kwargs = {}
    if format in ("JPEG", "JPG"):
        save_kwargs = {"quality": quality, "optimize": True}
    elif format == "WEBP":
        save_kwargs = {"quality": quality}
    image.save(buf, format=format, **save_kwargs)
    return buf.getvalue()
