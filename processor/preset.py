"""预设管理：保存、加载、列出、删除处理预设"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from .pipeline import Pipeline

import sys
from pathlib import Path


def _get_preset_dir() -> Path:
    """获取预设目录"""
    base = Path(os.environ.get("IMAGE_PIPELINE_HOME", str(Path.home() / ".image_pipeline")))
    preset_dir = base / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    return preset_dir


def save_preset(
    name: str,
    pipeline: Pipeline,
    description: str = "",
    overwrite: bool = True,
    preset_dir: Optional[Path] = None,
) -> str:
    """保存预设
    
    Args:
        name: 预设名称
        pipeline: 流水线配置
        description: 描述
        overwrite: 是否覆盖同名预设
        preset_dir: 自定义预设目录
    
    Returns:
        预设文件路径
    """
    pd = Path(preset_dir) if preset_dir else _get_preset_dir()
    safe_name = _sanitize_name(name)
    file_path = pd / f"{safe_name}.json"
    
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"预设 '{name}' 已存在")
    
    data = {
        "name": name,
        "description": description,
        "pipeline": pipeline.to_dict(),
        "version": 1,
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return str(file_path)


def load_preset(
    name: str,
    preset_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """加载预设
    
    Returns:
        {name, description, pipeline} 或 None
    """
    pd = Path(preset_dir) if preset_dir else _get_preset_dir()
    safe_name = _sanitize_name(name)
    file_path = pd / f"{safe_name}.json"
    
    if not file_path.exists():
        for f in pd.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("name") == name:
                    data["pipeline"] = Pipeline.from_dict(data["pipeline"])
                    return data
            except Exception:
                continue
        return None
    
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        data["pipeline"] = Pipeline.from_dict(data["pipeline"])
        return data
    except Exception:
        return None


def list_presets(preset_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """列出所有预设"""
    pd = Path(preset_dir) if preset_dir else _get_preset_dir()
    presets = []
    
    for f in sorted(pd.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            presets.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "step_count": len(data.get("pipeline", {}).get("steps", [])),
                "file_path": str(f),
            })
        except Exception:
            continue
    
    return presets


def delete_preset(name: str, preset_dir: Optional[Path] = None) -> bool:
    """删除预设
    
    Returns:
        是否成功删除
    """
    pd = Path(preset_dir) if preset_dir else _get_preset_dir()
    safe_name = _sanitize_name(name)
    file_path = pd / f"{safe_name}.json"
    
    if file_path.exists():
        file_path.unlink()
        return True
    
    for f in pd.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("name") == name:
                f.unlink()
                return True
        except Exception:
            continue
    
    return False


def _sanitize_name(name: str) -> str:
    """清理文件名"""
    import re
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:100] or "unnamed"


_BUILTIN_PRESETS = {
    "instagram_default": {
        "name": "Instagram 默认风",
        "description": "高对比+高饱和，适合社交平台分享",
        "pipeline": {
            "name": "Instagram 默认风",
            "steps": [
                {"step_type": "wb_auto", "enabled": True, "params": {}},
                {"step_type": "basic_adjust", "enabled": True, "params": {
                    "contrast": 15, "saturation": 12, "vibrance": 8,
                    "highlights": -10, "shadows": 15,
                }},
                {"step_type": "curves", "enabled": True, "params": {"brightness": 0, "contrast": 0}},
                {"step_type": "resize_long_edge", "enabled": True, "params": {"max_long_edge": 2048}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "JPEG", "quality": 90, "keep_exif": False,
                }},
            ],
        },
    },
    "portrait_soft": {
        "name": "人像柔光",
        "description": "柔和色调，暖调肤色，适合人像照片",
        "pipeline": {
            "name": "人像柔光",
            "steps": [
                {"step_type": "crop_smart", "enabled": True, "params": {"mode": "face", "ratio": 1.33}},
                {"step_type": "wb_auto", "enabled": True, "params": {}},
                {"step_type": "temp_tint", "enabled": True, "params": {"temperature": 10, "tint": -5}},
                {"step_type": "basic_adjust", "enabled": True, "params": {
                    "brightness": 5, "contrast": 8, "saturation": -5, "vibrance": 15,
                    "shadows": 20, "highlights": -5,
                }},
                {"step_type": "skin_smooth_lite", "enabled": False, "params": {}},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"auto_strength": True}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "JPEG", "quality": 95, "keep_exif": True,
                }},
            ],
        },
    },
    "landscape_hdr": {
        "name": "风景HDR",
        "description": "提升动态范围，鲜艳色彩，适合风光摄影",
        "pipeline": {
            "name": "风景HDR",
            "steps": [
                {"step_type": "crop_smart", "enabled": True, "params": {"mode": "rule_of_thirds", "ratio": 1.5}},
                {"step_type": "wb_auto", "enabled": True, "params": {}},
                {"step_type": "basic_adjust", "enabled": True, "params": {
                    "contrast": 20, "saturation": 18, "vibrance": 15,
                    "highlights": -25, "shadows": 30, "whites": -10, "blacks": 10,
                }},
                {"step_type": "lut_filter", "enabled": True, "params": {"preset_name": "cinematic_teal"}},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"auto_strength": True}},
                {"step_type": "resize_long_edge", "enabled": True, "params": {"max_long_edge": 4096}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "TIFF", "quality": 95, "keep_exif": True,
                }},
            ],
        },
    },
    "old_photo_restore": {
        "name": "老照片修复",
        "description": "划痕修复、去噪、色彩校正、对比度增强",
        "pipeline": {
            "name": "老照片修复",
            "steps": [
                {"step_type": "restore_photo", "enabled": True, "params": {
                    "remove_scratches": True, "denoise": True,
                    "color_correct": True, "enhance_contrast": True,
                }},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"auto_strength": True}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "TIFF", "quality": 95, "keep_exif": False,
                }},
            ],
        },
    },
    "ecommerce_square": {
        "name": "电商白底方图",
        "description": "去背景+智能填充白底方图，尺寸1080x1080",
        "pipeline": {
            "name": "电商白底方图",
            "steps": [
                {"step_type": "remove_bg", "enabled": True, "params": {
                    "method": "auto", "replace_with_color": True, "bg_color": "#ffffff",
                }},
                {"step_type": "smart_pad", "enabled": True, "params": {
                    "width": 1080, "height": 1080, "mode": "color", "bg_color": "#ffffff",
                }},
                {"step_type": "basic_adjust", "enabled": True, "params": {
                    "brightness": 5, "contrast": 10, "saturation": 5,
                }},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"auto_strength": True}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "JPEG", "quality": 92, "keep_exif": False,
                }},
            ],
        },
    },
    "web_optimized": {
        "name": "网页优化",
        "description": "WebP格式，长边2048，高质量压缩，剥离EXIF",
        "pipeline": {
            "name": "网页优化",
            "steps": [
                {"step_type": "resize_long_edge", "enabled": True, "params": {"max_long_edge": 2048}},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"auto_strength": True}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "WEBP", "quality": 85, "optimize": True, "keep_exif": False,
                }},
            ],
        },
    },
    "print_ready": {
        "name": "打印输出",
        "description": "300DPI适配，TIFF无损格式，统一24MP分辨率",
        "pipeline": {
            "name": "打印输出",
            "steps": [
                {"step_type": "wb_auto", "enabled": True, "params": {}},
                {"step_type": "basic_adjust", "enabled": True, "params": {
                    "contrast": 5, "saturation": 5,
                }},
                {"step_type": "uniform_resolution", "enabled": True, "params": {"target_mp": 24.0}},
                {"step_type": "smart_sharpen", "enabled": True, "params": {"strength": 1.2, "auto_strength": False}},
                {"step_type": "format_convert", "enabled": True, "params": {
                    "target_format": "TIFF", "quality": 100, "optimize": True, "keep_exif": True,
                }},
            ],
        },
    },
}


def ensure_builtin_presets(preset_dir: Optional[Path] = None) -> int:
    """确保内置预设存在（首次运行时创建）
    
    Returns:
        实际创建的预设数量
    """
    pd = Path(preset_dir) if preset_dir else _get_preset_dir()
    created = 0
    
    existing = list_presets(pd)
    existing_names = {p["name"] for p in existing}
    
    for key, data in _BUILTIN_PRESETS.items():
        if data["name"] in existing_names:
            continue
        try:
            pipeline = Pipeline.from_dict(data["pipeline"])
            save_preset(data["name"], pipeline, data["description"], True, pd)
            created += 1
        except Exception:
            pass
    
    return created
