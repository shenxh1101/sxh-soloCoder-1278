"""智能图片批处理流水线核心处理器包"""

from .core import (
    load_image,
    save_image,
    pil_to_cv,
    cv_to_pil,
    read_exif,
    write_exif,
    strip_exif,
    get_image_info,
    list_images_in_dir,
    extract_zip,
    create_zip,
    generate_thumbnail,
)

from .pipeline import (
    Pipeline,
    PipelineStep,
    PipelineResult,
    STEP_REGISTRY,
    register_step,
    execute_pipeline,
    batch_execute_pipeline,
)

from .preset import (
    save_preset,
    load_preset,
    list_presets,
    delete_preset,
)

__version__ = "1.0.0"
__all__ = [
    "load_image", "save_image", "pil_to_cv", "cv_to_pil",
    "read_exif", "write_exif", "strip_exif", "get_image_info",
    "list_images_in_dir", "extract_zip", "create_zip", "generate_thumbnail",
    "Pipeline", "PipelineStep", "PipelineResult",
    "STEP_REGISTRY", "register_step", "execute_pipeline", "batch_execute_pipeline",
    "save_preset", "load_preset", "list_presets", "delete_preset",
]
