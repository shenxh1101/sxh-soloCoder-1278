"""智能图片批处理流水线核心处理器包"""

import importlib

_LAZY_MODULES = {
    "core": "processor.core",
    "crop": "processor.crop",
    "color": "processor.color",
    "resize": "processor.resize",
    "format_rename": "processor.format_rename",
    "ai_enhance": "processor.ai_enhance",
    "pipeline": "processor.pipeline",
    "preset": "processor.preset",
}

_PUBLIC_NAMES = {
    "load_image": ("core", "load_image"),
    "save_image": ("core", "save_image"),
    "pil_to_cv": ("core", "pil_to_cv"),
    "cv_to_pil": ("core", "cv_to_pil"),
    "read_exif": ("core", "read_exif"),
    "write_exif": ("core", "write_exif"),
    "strip_exif": ("core", "strip_exif"),
    "get_image_info": ("core", "get_image_info"),
    "list_images_in_dir": ("core", "list_images_in_dir"),
    "extract_zip": ("core", "extract_zip"),
    "create_zip": ("core", "create_zip"),
    "generate_thumbnail": ("core", "generate_thumbnail"),
    "image_to_bytes": ("core", "image_to_bytes"),
    "Pipeline": ("pipeline", "Pipeline"),
    "PipelineStep": ("pipeline", "PipelineStep"),
    "PipelineResult": ("pipeline", "PipelineResult"),
    "STEP_REGISTRY": ("pipeline", "STEP_REGISTRY"),
    "register_step": ("pipeline", "register_step"),
    "execute_pipeline": ("pipeline", "execute_pipeline"),
    "batch_execute_pipeline": ("pipeline", "batch_execute_pipeline"),
    "get_step_categories": ("pipeline", "get_step_categories"),
    "save_preset": ("preset", "save_preset"),
    "load_preset": ("preset", "load_preset"),
    "list_presets": ("preset", "list_presets"),
    "delete_preset": ("preset", "delete_preset"),
    "ensure_builtin_presets": ("preset", "ensure_builtin_presets"),
}

_cache = {}


def __getattr__(name):
    if name in _PUBLIC_NAMES:
        mod_name, attr_name = _PUBLIC_NAMES[name]
        if mod_name not in _cache:
            try:
                _cache[mod_name] = importlib.import_module(_LAZY_MODULES[mod_name])
            except Exception as e:
                raise ImportError(f"无法加载模块 {mod_name}: {e}") from e
        return getattr(_cache[mod_name], attr_name)
    raise AttributeError(f"模块 'processor' 没有属性 '{name}'")


def _ensure_module(mod_name: str):
    if mod_name not in _cache:
        try:
            _cache[mod_name] = importlib.import_module(_LAZY_MODULES[mod_name])
        except Exception as e:
            raise ImportError(f"无法加载模块 {mod_name}: {e}") from e
    return _cache[mod_name]


__version__ = "1.0.0"
