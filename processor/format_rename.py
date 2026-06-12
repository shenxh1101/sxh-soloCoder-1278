"""格式转换和重命名模块"""

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field

from PIL import Image

from .core import (
    save_image,
    strip_exif,
    read_exif,
    get_image_info,
)


FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tiff",
    "BMP": ".bmp",
    "HEIF": ".heic",
    "HEIC": ".heic",
    "AVIF": ".avif",
    "GIF": ".gif",
}


def get_format_extension(format_name: str) -> str:
    """获取格式对应的扩展名"""
    format_name = format_name.upper()
    return FORMAT_EXTENSIONS.get(format_name, ".jpg")


def convert_format(
    image: Image.Image,
    target_format: str,
    quality: int = 95,
    optimize: bool = True,
    keep_exif: bool = True,
) -> Image.Image:
    """转换图像格式（在内存中处理，用于流水线）
    
    Args:
        image: 输入图像
        target_format: 目标格式 - JPEG, PNG, WEBP, TIFF, BMP, HEIF
        quality: 压缩质量 (1-100)
        optimize: 是否优化
        keep_exif: 是否保留EXIF信息
    
    Returns:
        返回处理后的图像对象（包含目标格式的元数据标记）
    """
    target_format = target_format.upper()
    result = image.copy()
    
    if not keep_exif:
        result = strip_exif(result)
    
    if target_format in ("JPEG", "JPG"):
        if result.mode in ("RGBA", "LA", "P"):
            result = result.convert("RGB")
    elif target_format == "PNG":
        pass
    elif target_format == "WEBP":
        pass
    elif target_format in ("TIFF", "TIF"):
        pass
    elif target_format in ("HEIC", "HEIF"):
        if result.mode == "RGBA":
            result = result.convert("RGB")
    
    result._format_hint = target_format
    result._save_kwargs = {
        "quality": quality,
        "optimize": optimize,
        "keep_exif": keep_exif,
        "format": target_format,
    }
    
    return result


def convert_and_save(
    image: Image.Image,
    output_path: Union[str, Path],
    target_format: Optional[str] = None,
    quality: int = 95,
    optimize: bool = True,
    keep_exif: bool = True,
) -> str:
    """转换格式并保存
    
    Returns:
        实际保存的文件路径
    """
    output_path = Path(output_path)
    
    if target_format is not None:
        target_format = target_format.upper()
        ext = get_format_extension(target_format)
        output_path = output_path.with_suffix(ext)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_image(
        image,
        str(output_path),
        format=target_format,
        quality=quality,
        optimize=optimize,
        keep_exif=keep_exif,
    )
    
    return str(output_path)


@dataclass
class RenameContext:
    original_path: str
    original_name: str
    original_stem: str
    original_ext: str
    index: int
    total: int
    exif: Dict[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0


def _parse_gps_from_exif(exif: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """从EXIF GPS IFD直接解析经纬度（度分秒格式）"""
    try:
        lat_val = exif.get("GPS.GPSLatitude")
        lat_ref = exif.get("GPS.GPSLatitudeRef")
        lon_val = exif.get("GPS.GPSLongitude")
        lon_ref = exif.get("GPS.GPSLongitudeRef")
        
        if lat_val is None or lon_val is None:
            return None
        
        def dms_to_decimal(dms, ref):
            if isinstance(dms, (tuple, list)) and len(dms) == 3:
                d, m, s = dms
                if isinstance(d, tuple) and len(d) == 2:
                    d = d[0] / d[1] if d[1] else 0
                if isinstance(m, tuple) and len(m) == 2:
                    m = m[0] / m[1] if m[1] else 0
                if isinstance(s, tuple) and len(s) == 2:
                    s = s[0] / s[1] if s[1] else 0
                decimal = float(d) + float(m) / 60.0 + float(s) / 3600.0
                if isinstance(ref, bytes):
                    ref = ref.decode("utf-8", errors="ignore")
                if str(ref).upper() in ("S", "W"):
                    decimal = -decimal
                return decimal
            elif isinstance(dms, (int, float)):
                return float(dms)
            return None
        
        lat = dms_to_decimal(lat_val, lat_ref)
        lon = dms_to_decimal(lon_val, lon_ref)
        if lat is not None and lon is not None:
            return (lat, lon)
    except Exception:
        pass
    return None


def _format_gps_dms(lat: float, lon: float) -> str:
    """将十进制经纬度转换为度分秒可读格式"""
    def to_dms(decimal, is_lat):
        direction = ("N" if is_lat else "E") if decimal >= 0 else ("S" if is_lat else "W")
        decimal = abs(decimal)
        d = int(decimal)
        m = int((decimal - d) * 60)
        s = (decimal - d - m / 60.0) * 3600.0
        return f"{d}{direction}{m:02d}{s:02.0f}"
    return f"{to_dms(lat, True)}_{to_dms(lon, False)}"


def _reverse_geocode_simple(lat: float, lon: float) -> str:
    """简易逆地理编码：返回纬度带+经度带作为可读位置片段（无需网络）"""
    def lat_zone(lat_v):
        if lat_v > 66.5: return "Arctic"
        elif lat_v > 55: return "North"
        elif lat_v > 45: return "NMid"
        elif lat_v > 35: return "Mediter"
        elif lat_v > 23.5: return "SubTrop"
        elif lat_v > 10: return "Tropic"
        elif lat_v > -10: return "Equator"
        elif lat_v > -23.5: return "STropic"
        elif lat_v > -35: return "SSubTrop"
        elif lat_v > -55: return "SMid"
        else: return "Antarc"
    
    def lon_zone(lon_v):
        if lon_v > 150: return "WPac"
        elif lon_v > 120: return "EAsia"
        elif lon_v > 90: return "SAsia"
        elif lon_v > 60: return "Mideast"
        elif lon_v > 30: return "EAfrica"
        elif lon_v > 0: return "WEurope"
        elif lon_v > -30: return "WAtlantic"
        elif lon_v > -60: return "SAmerica"
        elif lon_v > -90: return "CAmerica"
        else: return "Pacific"
    
    return f"{lat_zone(lat)}_{lon_zone(lon)}"


def _get_camera_model(exif: Dict[str, Any]) -> str:
    """从EXIF获取相机型号"""
    model = exif.get("0th.Model") or exif.get("IFD0.Model") or exif.get("Model") or ""
    if isinstance(model, bytes):
        model = model.decode("utf-8", errors="ignore")
    return str(model).strip().replace(" ", "_").replace("/", "_")


def _get_lens_model(exif: Dict[str, Any]) -> str:
    """从EXIF获取镜头型号"""
    lens = (
        exif.get("Exif.LensModel")
        or exif.get("Exif.LensMake")
        or exif.get("LensModel")
        or ""
    )
    if isinstance(lens, bytes):
        lens = lens.decode("utf-8", errors="ignore")
    return str(lens).strip().replace(" ", "_").replace("/", "_")


def _get_shoot_date(exif: Dict[str, Any]) -> Optional[datetime]:
    """从EXIF获取拍摄时间"""
    date_keys = [
        "Exif.DateTimeOriginal",
        "Exif.DateTimeDigitized",
        "0th.DateTime",
        "IFD0.DateTime",
        "DateTime",
    ]
    for key in date_keys:
        value = exif.get(key)
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            value = str(value).strip()
            for fmt in [
                "%Y:%m:%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y:%m:%d",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
    return None


def _sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name[:200] if len(name) > 200 else name


def generate_new_name(
    context: RenameContext,
    template: str = "{stem}_{index:04d}",
) -> str:
    """根据模板生成新文件名
    
    支持的模板变量：
    {stem} - 原文件名（不含扩展名）
    {ext} - 原扩展名
    {index} - 序号
    {index02d}, {index03d}, {index04d} - 带前导零的序号
    {total} - 总数
    {date} - 拍摄日期 (YYYYMMDD)
    {datetime} - 拍摄时间 (YYYYMMDD_HHMMSS)
    {year} - 年
    {month} - 月
    {day} - 日
    {hour} - 时
    {minute} - 分
    {second} - 秒
    {camera} - 相机型号
    {lens} - 镜头型号
    {width} - 图像宽度
    {height} - 图像高度
    {size} - WxH尺寸
    {focal} - 焦距
    {aperture} - 光圈
    {shutter} - 快门
    {iso} - ISO
    
    Args:
        context: 重命名上下文
        template: 命名模板
    
    Returns:
        新文件名（不含扩展名）
    """
    shoot_date = _get_shoot_date(context.exif)
    
    if shoot_date:
        date_str = shoot_date.strftime("%Y%m%d")
        datetime_str = shoot_date.strftime("%Y%m%d_%H%M%S")
        year = shoot_date.strftime("%Y")
        month = shoot_date.strftime("%m")
        day = shoot_date.strftime("%d")
        hour = shoot_date.strftime("%H")
        minute = shoot_date.strftime("%M")
        second = shoot_date.strftime("%S")
    else:
        try:
            mtime = datetime.fromtimestamp(Path(context.original_path).stat().st_mtime)
        except Exception:
            mtime = datetime.now()
        date_str = mtime.strftime("%Y%m%d")
        datetime_str = mtime.strftime("%Y%m%d_%H%M%S")
        year = mtime.strftime("%Y")
        month = mtime.strftime("%m")
        day = mtime.strftime("%d")
        hour = mtime.strftime("%H")
        minute = mtime.strftime("%M")
        second = mtime.strftime("%S")
    
    camera = _get_camera_model(context.exif) or "NoCamera"
    lens = _get_lens_model(context.exif) or "NoLens"
    
    gps = _parse_gps_from_exif(context.exif)
    if gps:
        gps_lat = f"{gps[0]:.4f}"
        gps_lon = f"{gps[1]:.4f}"
        gps_dms = _format_gps_dms(gps[0], gps[1])
        location = _reverse_geocode_simple(gps[0], gps[1])
        gps_short = f"{abs(gps[0]):.0f}{('N' if gps[0]>=0 else 'S')}{abs(gps[1]):.0f}{('E' if gps[1]>=0 else 'W')}"
    else:
        gps_lat = "NoGPS"
        gps_lon = "NoGPS"
        gps_dms = "NoGPS"
        location = "NoLoc"
        gps_short = "NoGPS"
    
    focal = context.exif.get("Exif.FocalLength") or context.exif.get("FocalLength") or ""
    aperture = context.exif.get("Exif.FNumber") or context.exif.get("FNumber") or ""
    shutter = context.exif.get("Exif.ExposureTime") or context.exif.get("ExposureTime") or ""
    iso = context.exif.get("Exif.ISOSpeedRatings") or context.exif.get("ISOSpeedRatings") or ""
    
    if isinstance(focal, tuple) and len(focal) == 2:
        focal = f"{focal[0] / focal[1]:.0f}mm" if focal[1] else ""
    elif focal:
        focal = f"{focal}mm"
    else:
        focal = "NoFL"
    
    if isinstance(aperture, tuple) and len(aperture) == 2:
        aperture = f"f{aperture[0] / aperture[1]:.1f}" if aperture[1] else ""
    elif aperture:
        aperture = f"f{aperture}"
    else:
        aperture = "NoF"
    
    if isinstance(shutter, tuple) and len(shutter) == 2:
        if shutter[1] and shutter[0] < shutter[1]:
            shutter = f"1{shutter[1] // shutter[0]}s" if shutter[0] else ""
        elif shutter[1]:
            shutter = f"{shutter[0] / shutter[1]:.2f}s"
        else:
            shutter = ""
    elif shutter:
        shutter = f"{shutter}s"
    else:
        shutter = "NoSS"
    
    if not iso:
        iso = "NoISO"
    
    try:
        result = template.format(
            stem=context.original_stem,
            ext=context.original_ext,
            index=context.index,
            index02d=f"{context.index:02d}",
            index03d=f"{context.index:03d}",
            index04d=f"{context.index:04d}",
            index05d=f"{context.index:05d}",
            total=context.total,
            date=date_str,
            datetime=datetime_str,
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            camera=camera,
            lens=lens,
            width=context.width,
            height=context.height,
            size=f"{context.width}x{context.height}",
            focal=str(focal),
            aperture=str(aperture),
            shutter=str(shutter),
            iso=str(iso),
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_dms=gps_dms,
            location=location,
            gps_short=gps_short,
        )
    except (KeyError, IndexError, ValueError):
        result = f"{context.original_stem}_{context.index:04d}"
    
    return _sanitize_filename(result)


def batch_rename(
    file_paths: list,
    template: str = "{stem}_{index:04d}",
    start_index: int = 1,
) -> Dict[str, str]:
    """批量生成重命名映射
    
    Returns:
        {原路径: 新文件名（含扩展名）}
    """
    rename_map = {}
    total = len(file_paths)
    
    for idx, path in enumerate(file_paths, start_index):
        path_obj = Path(path)
        info = get_image_info(path)
        
        ctx = RenameContext(
            original_path=path,
            original_name=path_obj.name,
            original_stem=path_obj.stem,
            original_ext=path_obj.suffix.lstrip("."),
            index=idx,
            total=total,
            exif=info.exif,
            width=info.width,
            height=info.height,
        )
        
        new_name = generate_new_name(ctx, template)
        new_fullname = f"{new_name}{path_obj.suffix.lower()}"
        rename_map[path] = new_fullname
    
    return rename_map
