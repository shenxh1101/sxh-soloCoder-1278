"""流水线引擎：步骤注册、单步执行、流水线执行、多进程批量处理"""

import os
import sys
import time
import json
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from PIL import Image

from .core import load_image, save_image, read_exif, get_image_info
from .crop import smart_crop
from .color import (
    adjust_white_balance_auto,
    adjust_temperature_tint,
    adjust_curves,
    adjust_hsl,
    adjust_color_balance,
    apply_vibrance,
    apply_preset_lut,
    adjust_basic,
    list_available_luts,
)
from .resize import (
    resize_by_scale,
    resize_to_width,
    resize_to_height,
    resize_long_edge,
    resize_short_edge,
    resize_exact,
    resize_fit,
    smart_pad_to_size,
    resize_uniform_resolution,
    add_border,
)
from .format_rename import (
    convert_format,
    batch_rename,
    generate_new_name,
    RenameContext,
)
from .ai_enhance import (
    remove_background,
    smart_sharpen,
    unsharp_mask,
    restore_old_photo,
    apply_style_transfer,
    list_available_styles,
)


STEP_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_step(
    step_type: str,
    name: str,
    description: str,
    default_params: Dict[str, Any],
    category: str = "通用",
):
    """注册处理步骤装饰器/函数"""
    
    def decorator(func: Callable) -> Callable:
        STEP_REGISTRY[step_type] = {
            "func": func,
            "name": name,
            "description": description,
            "default_params": default_params,
            "category": category,
        }
        return func
    
    return decorator


@dataclass
class PipelineStep:
    step_type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "enabled": self.enabled,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        return cls(
            step_type=data["step_type"],
            enabled=data.get("enabled", True),
            params=data.get("params", {}),
        )


@dataclass
class Pipeline:
    steps: List[PipelineStep] = field(default_factory=list)
    name: str = "未命名流水线"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        return cls(
            name=data.get("name", "未命名流水线"),
            steps=[PipelineStep.from_dict(s) for s in data.get("steps", [])],
        )


@dataclass
class PipelineResult:
    source_path: str
    output_path: Optional[str] = None
    new_filename: Optional[str] = None
    success: bool = True
    error_message: str = ""
    processing_time: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@register_step("crop_smart", "智能裁剪", "基于人脸/主体/三分法的自动构图裁剪", {
    "mode": "rule_of_thirds",
    "ratio": None,
}, "裁剪")
def step_crop_smart(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    mode = params.get("mode", "rule_of_thirds")
    ratio = params.get("ratio")
    return smart_crop(image, mode=mode, ratio=ratio)


@register_step("crop_fixed", "固定比例裁剪", "按指定宽高比裁剪", {
    "ratio": 1.0,
    "mode": "center",
}, "裁剪")
def step_crop_fixed(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    ratio = params.get("ratio", 1.0)
    mode = params.get("mode", "center")
    from .crop import crop_fixed_ratio
    return crop_fixed_ratio(image, ratio=ratio, mode=mode)


@register_step("wb_auto", "自动白平衡", "自动校正偏色", {}, "颜色")
def step_wb_auto(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return adjust_white_balance_auto(image)


@register_step("temp_tint", "色温色调", "调整色温(冷暖)和色调(绿品)", {
    "temperature": 0,
    "tint": 0,
}, "颜色")
def step_temp_tint(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    temperature = float(params.get("temperature", 0))
    tint = float(params.get("tint", 0))
    return adjust_temperature_tint(image, temperature, tint)


@register_step("basic_adjust", "基础调色", "曝光/亮度/对比/高光/阴影/饱和度", {
    "exposure": 0,
    "brightness": 0,
    "contrast": 0,
    "highlights": 0,
    "shadows": 0,
    "whites": 0,
    "blacks": 0,
    "saturation": 0,
    "vibrance": 0,
}, "颜色")
def step_basic_adjust(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return adjust_basic(
        image,
        exposure=float(params.get("exposure", 0)),
        brightness=float(params.get("brightness", 0)),
        contrast=float(params.get("contrast", 0)),
        highlights=float(params.get("highlights", 0)),
        shadows=float(params.get("shadows", 0)),
        whites=float(params.get("whites", 0)),
        blacks=float(params.get("blacks", 0)),
        saturation=float(params.get("saturation", 0)),
        vibrance=float(params.get("vibrance", 0)),
    )


@register_step("hsl", "HSL调节", "色相/饱和度/明度单独调节", {
    "hue": 0,
    "saturation": 0,
    "lightness": 0,
}, "颜色")
def step_hsl(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return adjust_hsl(
        image,
        hue=float(params.get("hue", 0)),
        saturation=float(params.get("saturation", 0)),
        lightness=float(params.get("lightness", 0)),
    )


@register_step("color_balance", "色彩平衡", "阴影/中间调/高光单独调色", {
    "shadows_cr": 0, "shadows_mg": 0, "shadows_yb": 0,
    "midtones_cr": 0, "midtones_mg": 0, "midtones_yb": 0,
    "highlights_cr": 0, "highlights_mg": 0, "highlights_yb": 0,
}, "颜色")
def step_color_balance(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return adjust_color_balance(
        image,
        shadows=(int(params.get("shadows_cr", 0)), int(params.get("shadows_mg", 0)), int(params.get("shadows_yb", 0))),
        midtones=(int(params.get("midtones_cr", 0)), int(params.get("midtones_mg", 0)), int(params.get("midtones_yb", 0))),
        highlights=(int(params.get("highlights_cr", 0)), int(params.get("highlights_mg", 0)), int(params.get("highlights_yb", 0))),
    )


@register_step("lut_filter", "LUT滤镜", "应用预设LUT滤镜/风格滤镜", {
    "preset_name": "vintage",
}, "颜色")
def step_lut_filter(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    preset = params.get("preset_name", "vintage")
    return apply_preset_lut(image, preset)


@register_step("curves", "色调曲线", "曲线控制点调整", {
    "brightness": 0,
    "contrast": 0,
}, "颜色")
def step_curves(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    points = params.get("points")
    brightness = float(params.get("brightness", 0))
    contrast = float(params.get("contrast", 0))
    return adjust_curves(image, "rgb", points, brightness, contrast)


@register_step("resize_scale", "比例缩放", "按比例缩放图像", {
    "scale": 1.0,
}, "尺寸")
def step_resize_scale(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    scale = float(params.get("scale", 1.0))
    return resize_by_scale(image, scale)


@register_step("resize_fit", "适配缩放", "等比缩放以适应边界框", {
    "width": 1920,
    "height": 1080,
}, "尺寸")
def step_resize_fit(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    max_w = int(params.get("width", 1920))
    max_h = int(params.get("height", 1080))
    return resize_fit(image, (max_w, max_h))


@register_step("resize_long_edge", "限制长边", "限制图像最长边尺寸", {
    "max_long_edge": 2048,
}, "尺寸")
def step_resize_long_edge(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    max_le = int(params.get("max_long_edge", 2048))
    return resize_long_edge(image, max_le)


@register_step("resize_short_edge", "限制短边", "限制图像最短边尺寸", {
    "min_short_edge": 1024,
}, "尺寸")
def step_resize_short_edge(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    min_se = int(params.get("min_short_edge", 1024))
    return resize_short_edge(image, min_se)


@register_step("resize_exact", "精确尺寸", "强制缩放到精确尺寸", {
    "width": 800,
    "height": 600,
    "allow_upscale": True,
}, "尺寸")
def step_resize_exact(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    w = int(params.get("width", 800))
    h = int(params.get("height", 600))
    allow_up = bool(params.get("allow_upscale", True))
    return resize_exact(image, (w, h), allow_upscale=allow_up)


@register_step("smart_pad", "智能填充", "缩放+填充背景到指定尺寸", {
    "width": 1080,
    "height": 1080,
    "mode": "auto",
    "bg_color": "#ffffff",
}, "尺寸")
def step_smart_pad(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    w = int(params.get("width", 1080))
    h = int(params.get("height", 1080))
    mode = params.get("mode", "auto")
    color_str = params.get("bg_color", "#ffffff")
    bg_color = None
    if isinstance(color_str, str) and color_str.startswith("#"):
        bg_color = tuple(int(color_str.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    return smart_pad_to_size(image, (w, h), mode=mode, bg_color=bg_color)


@register_step("uniform_resolution", "统一分辨率", "统一图像到指定百万像素数", {
    "target_mp": 24.0,
}, "尺寸")
def step_uniform_resolution(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    target_mp = float(params.get("target_mp", 24.0))
    return resize_uniform_resolution(image, target_mp)


@register_step("add_border", "添加边框", "为图像添加边框", {
    "border_width": 20,
    "color": "#ffffff",
}, "尺寸")
def step_add_border(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    bw = int(params.get("border_width", 20))
    color_str = params.get("color", "#ffffff")
    color = tuple(int(color_str.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    return add_border(image, bw, color)


@register_step("remove_bg", "自动去背景", "AI/显著性检测自动去除背景", {
    "method": "auto",
    "replace_with_color": False,
    "bg_color": "#ffffff",
}, "AI")
def step_remove_bg(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    method = params.get("method", "auto")
    replace = bool(params.get("replace_with_color", False))
    bg_color = None
    if replace:
        color_str = params.get("bg_color", "#ffffff")
        bg_color = tuple(int(color_str.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    return remove_background(image, method=method, bg_color=bg_color)


@register_step("smart_sharpen", "智能锐化", "自动检测模糊程度并锐化/去模糊", {
    "auto_strength": True,
    "strength": 1.0,
}, "AI")
def step_smart_sharpen(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    auto = bool(params.get("auto_strength", True))
    if auto:
        return smart_sharpen(image, strength=None)
    else:
        strength = float(params.get("strength", 1.0))
        return smart_sharpen(image, strength=strength)


@register_step("unsharp_mask", "USM锐化", "标准USM锐化滤镜", {
    "amount": 1.0,
    "radius": 1.5,
    "threshold": 2,
}, "AI")
def step_unsharp_mask(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return unsharp_mask(
        image,
        amount=float(params.get("amount", 1.0)),
        radius=float(params.get("radius", 1.5)),
        threshold=int(params.get("threshold", 2)),
    )


@register_step("restore_photo", "老照片修复", "划痕检测修复+去噪+色彩校正", {
    "remove_scratches": True,
    "denoise": True,
    "color_correct": True,
    "enhance_contrast": True,
}, "AI")
def step_restore_photo(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return restore_old_photo(
        image,
        remove_scratches=bool(params.get("remove_scratches", True)),
        denoise=bool(params.get("denoise", True)),
        color_correct=bool(params.get("color_correct", True)),
        enhance_contrast=bool(params.get("enhance_contrast", True)),
    )


@register_step("style_transfer", "风格迁移", "转换为油画/水彩/素描/电影等风格", {
    "style": "oil",
}, "AI")
def step_style_transfer(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    style = params.get("style", "oil")
    return apply_style_transfer(image, style)


@register_step("format_convert", "格式转换", "转换图像格式并调整压缩质量", {
    "target_format": "JPEG",
    "quality": 95,
    "optimize": True,
    "keep_exif": True,
}, "输出")
def step_format_convert(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    return convert_format(
        image,
        target_format=params.get("target_format", "JPEG"),
        quality=int(params.get("quality", 95)),
        optimize=bool(params.get("optimize", True)),
        keep_exif=bool(params.get("keep_exif", True)),
    )


@register_step("strip_exif", "剥离EXIF", "移除图像中的元数据信息", {}, "输出")
def step_strip_exif(image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    from .core import strip_exif
    return strip_exif(image)


def execute_pipeline(
    image: Image.Image,
    pipeline: Pipeline,
    capture_intermediate: bool = False,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """执行流水线，返回 (最终图像, 各步骤结果)
    
    Args:
        image: 输入图像
        pipeline: 流水线配置
        capture_intermediate: 是否捕获每个步骤的中间结果图像
    
    Returns:
        (处理后图像, 步骤结果列表)
    """
    current = image
    step_results = []
    
    for step in pipeline.steps:
        if not step.enabled:
            step_results.append({
                "step_type": step.step_type,
                "enabled": False,
                "skipped": True,
                "time_ms": 0,
            })
            continue
        
        step_info = STEP_REGISTRY.get(step.step_type)
        if step_info is None:
            step_results.append({
                "step_type": step.step_type,
                "error": f"未知步骤类型: {step.step_type}",
                "skipped": True,
                "time_ms": 0,
            })
            continue
        
        func = step_info["func"]
        default_params = step_info["default_params"]
        
        effective_params = {**default_params, **step.params}
        
        start = time.time()
        try:
            current = func(current, effective_params)
            elapsed = (time.time() - start) * 1000
            
            result = {
                "step_type": step.step_type,
                "enabled": True,
                "success": True,
                "time_ms": round(elapsed, 2),
                "params": effective_params,
            }
            if capture_intermediate:
                result["image"] = current.copy()
            
            step_results.append(result)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            step_results.append({
                "step_type": step.step_type,
                "enabled": True,
                "success": False,
                "error": str(e),
                "time_ms": round(elapsed, 2),
                "traceback": traceback.format_exc(),
            })
    
    return current, step_results


def _process_single_image(args_tuple):
    """多进程工作函数：处理单张图片（顶层函数可被pickle）"""
    (
        source_path, output_dir, pipeline_dict, rename_map_entry,
        idx, total, input_root_dir,
    ) = args_tuple

    start = time.time()
    result = PipelineResult(source_path=source_path)

    try:
        pipeline = Pipeline.from_dict(pipeline_dict)
        image = load_image(source_path)

        processed, step_results = execute_pipeline(image, pipeline)

        new_filename = rename_map_entry if rename_map_entry else Path(source_path).name

        format_hint = getattr(processed, "_format_hint", None)
        save_kwargs = getattr(processed, "_save_kwargs", {})

        if format_hint:
            from .format_rename import get_format_extension
            ext = get_format_extension(format_hint)
            new_filename = str(Path(new_filename).with_suffix(ext))

        if input_root_dir:
            try:
                rel = Path(source_path).relative_to(Path(input_root_dir))
                rel_parent = rel.parent
                if str(rel_parent) != ".":
                    sub_out = Path(output_dir) / rel_parent
                    sub_out.mkdir(parents=True, exist_ok=True)
                    output_path = sub_out / new_filename
                else:
                    output_path = Path(output_dir) / new_filename
            except ValueError:
                output_path = Path(output_dir) / new_filename
        else:
            output_path = Path(output_dir) / new_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            stem = output_path.stem
            suffix = output_path.suffix
            counter = 1
            while True:
                candidate = output_path.with_name(f"{stem}_{counter:03d}{suffix}")
                if not candidate.exists():
                    output_path = candidate
                    new_filename = output_path.name
                    break
                counter += 1

        save_kwargs_final = {
            "quality": save_kwargs.get("quality", 95),
            "optimize": save_kwargs.get("optimize", True),
            "keep_exif": save_kwargs.get("keep_exif", True),
            "format": save_kwargs.get("format"),
        }

        save_image(processed, str(output_path), **save_kwargs_final)

        result.output_path = str(output_path)
        result.new_filename = str(Path(output_path).relative_to(output_dir)) if output_path.is_relative_to(output_dir) else output_path.name
        result.step_results = [
            {k: v for k, v in sr.items() if k != "image"}
            for sr in step_results
        ]
        result.processing_time = time.time() - start

    except Exception as e:
        result.success = False
        result.error_message = str(e) + "\n" + traceback.format_exc()
        result.processing_time = time.time() - start

    return idx, result


def batch_execute_pipeline(
    source_paths: List[str],
    output_dir: Union[str, Path],
    pipeline: Pipeline,
    rename_template: Optional[str] = None,
    num_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, PipelineResult], None]] = None,
    start_index: int = 1,
    input_root_dir: Optional[str] = None,
) -> List[PipelineResult]:
    """批量多进程执行流水线

    Args:
        source_paths: 源图片路径列表
        output_dir: 输出目录
        pipeline: 流水线配置
        rename_template: 重命名模板（None则使用原名）
        num_workers: 进程数，None=CPU核心数
        progress_callback: 回调(完成数量, 总数, 最新结果)
        start_index: 重命名起始序号
        input_root_dir: 源根目录，用于在输出中保留原目录层级

    Returns:
        处理结果列表（按输入顺序）
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total = len(source_paths)

    if rename_template:
        rename_map = batch_rename(source_paths, rename_template, start_index)
        rename_entries = [rename_map[p] for p in source_paths]
    else:
        rename_entries = [Path(p).name for p in source_paths]

    pipeline_dict = pipeline.to_dict()

    args_list = [
        (source_paths[i], output_dir, pipeline_dict, rename_entries[i], i, total, input_root_dir)
        for i in range(total)
    ]

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    num_workers = min(max(1, num_workers), total)

    results = [None] * total
    completed = 0

    if num_workers == 1 or total <= 2:
        for args in args_list:
            idx, res = _process_single_image(args)
            results[idx] = res
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total, res)
                except Exception:
                    pass
        return results

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        future_map = {
            executor.submit(_process_single_image, args): args[4]
            for args in args_list
        }

        for future in as_completed(future_map):
            idx, res = future.result()
            results[idx] = res
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total, res)
                except Exception:
                    pass

    return results


def get_step_categories() -> Dict[str, List[Dict[str, Any]]]:
    """获取按分类组织的步骤列表"""
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for step_type, info in STEP_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "step_type": step_type,
            "name": info["name"],
            "description": info["description"],
            "default_params": info["default_params"],
        })
    return categories
