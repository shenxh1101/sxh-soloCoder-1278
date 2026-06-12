"""智能图片批处理流水线 - Streamlit Web 界面

功能：
- 上传 ZIP 或选择本地目录
- 缩略图网格展示
- 可拖拽排序的处理步骤流水线
- 实时原图 vs 处理后对比预览
- 预设管理
- 多进程并行处理 + 进度显示
- 打包下载
"""

import os
import io
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="智能图片批处理流水线",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

_IMPORTS_OK = True
_IMPORT_ERRORS = []
_PIPELINE_READY = False

try:
    from processor.pipeline import (
        Pipeline, PipelineStep, PipelineResult,
        STEP_REGISTRY, get_step_categories,
        execute_pipeline, batch_execute_pipeline,
    )
    _PIPELINE_READY = True
except Exception as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"pipeline: {e}")
    STEP_REGISTRY = {}
    Pipeline = None
    PipelineStep = None

try:
    from processor.core import (
        load_image, save_image, generate_thumbnail,
        list_images_in_dir, extract_zip, create_zip,
        get_image_info, image_to_bytes,
    )
except Exception as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"core: {e}")
    load_image = None

try:
    from processor.preset import (
        save_preset, load_preset, list_presets,
        delete_preset, ensure_builtin_presets,
    )
except Exception as e:
    _IMPORT_ERRORS.append(f"preset: {e}")
    ensure_builtin_presets = lambda: 0
    list_presets = lambda: []
    save_preset = lambda *a, **k: ""
    load_preset = lambda *a, **k: None
    delete_preset = lambda *a, **k: False

try:
    from processor.color import list_available_luts
except Exception:
    list_available_luts = lambda: {}

try:
    from processor.ai_enhance import list_available_styles
except Exception:
    list_available_styles = lambda: {}

_HAS_SORTABLES = False
try:
    from streamlit_sortables import sorted_items
    _HAS_SORTABLES = True
except ImportError:
    pass


st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { margin-top: 0.5rem; }
    .step-card {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        background: #1a1a1a;
    }
    .step-card.enabled { border-left: 4px solid #4CAF50; }
    .step-card.disabled { opacity: 0.5; border-left: 4px solid #888; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #4CAF50, #81C784); }
    .tag {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 0 0.2rem;
    }
    .tag-crop { background: #E65100; color: white; }
    .tag-color { background: #1565C0; color: white; }
    .tag-size { background: #2E7D32; color: white; }
    .tag-ai { background: #6A1B9A; color: white; }
    .tag-output { background: #00695C; color: white; }
    .big-number { font-size: 1.8rem; font-weight: bold; text-align: center; }
    .metric-label { font-size: 0.85rem; opacity: 0.7; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助函数 (定义在使用之前)
# ============================================================

def _invalidate_preview_cache():
    st.session_state.preview_cache = {}


def _get_step_info(step_type: str) -> Dict[str, Any]:
    return STEP_REGISTRY.get(step_type, {
        "name": step_type,
        "default_params": {},
        "category": "未知",
        "description": "",
    })


def _render_step_params(idx: int, step, defaults: Dict[str, Any]):
    """渲染单个步骤的参数配置面板"""
    params = step.params
    t = step.step_type
    changed = False

    def _set(key, value):
        nonlocal changed
        if params.get(key) != value:
            params[key] = value
            changed = True

    if t == "crop_smart":
        modes = [("rule_of_thirds", "三分法构图(推荐)"), ("face", "人脸居中"),
                 ("subject", "主体检测居中"), ("fixed", "固定比例")]
        labels = [m[1] for m in modes]
        values = [m[0] for m in modes]
        cur = params.get("mode", "rule_of_thirds")
        sel = st.selectbox("裁剪模式", labels, index=values.index(cur) if cur in values else 0,
                          key=f"p_crop_m_{idx}")
        _set("mode", values[labels.index(sel)])
        ratios = [
            (None, "保持原图比例"), (1.0, "1:1 正方形"), (1.33, "4:3"),
            (1.5, "3:2"), (1.78, "16:9"), (0.75, "3:4"), (0.56, "9:16"),
        ]
        r_labels = [r[1] for r in ratios]
        r_values = [r[0] for r in ratios]
        cur_r = params.get("ratio")
        r_idx = 0
        for j, rv in enumerate(r_values):
            if rv is None and cur_r is None:
                r_idx = j; break
            elif rv is not None and cur_r is not None and abs(rv - cur_r) < 0.01:
                r_idx = j; break
        r_sel = st.selectbox("目标比例", r_labels, index=r_idx, key=f"p_crop_r_{idx}")
        _set("ratio", r_values[r_labels.index(r_sel)])

    elif t == "crop_fixed":
        r = st.number_input("宽高比 (宽/高)", 0.1, 10.0, float(params.get("ratio", 1.0)), 0.01,
                           key=f"p_cf_r_{idx}")
        _set("ratio", r)
        modes = ["center", "top", "bottom", "left", "right"]
        m_labels = ["居中", "顶部", "底部", "左侧", "右侧"]
        sel = st.selectbox("锚点位置", m_labels, index=modes.index(params.get("mode", "center")),
                          key=f"p_cf_m_{idx}")
        _set("mode", modes[m_labels.index(sel)])

    elif t == "temp_tint":
        _set("temperature", st.slider("色温 (冷蓝→暖黄)", -100, 100,
            int(params.get("temperature", 0)), key=f"p_tt_t_{idx}"))
        _set("tint", st.slider("色调 (绿→品红)", -100, 100,
            int(params.get("tint", 0)), key=f"p_tt_i_{idx}"))

    elif t == "basic_adjust":
        for k in ["exposure", "contrast", "brightness", "highlights",
                  "shadows", "whites", "blacks", "saturation", "vibrance"]:
            _set(k, st.slider(k, -100, 100, int(params.get(k, 0)),
                 key=f"p_ba_{k}_{idx}"))

    elif t == "hsl":
        _set("hue", st.slider("色相偏移", -180, 180, int(params.get("hue", 0)),
             key=f"p_hsl_h_{idx}"))
        _set("saturation", st.slider("饱和度", -100, 100, int(params.get("saturation", 0)),
             key=f"p_hsl_s_{idx}"))
        _set("lightness", st.slider("明度", -100, 100, int(params.get("lightness", 0)),
             key=f"p_hsl_l_{idx}"))

    elif t == "color_balance":
        st.markdown("**阴影** (青-红 / 洋红-绿 / 黄-蓝)")
        c1, c2, c3 = st.columns(3)
        _set("shadows_cr", c1.slider("S:青-红", -100, 100, int(params.get("shadows_cr", 0)),
             key=f"p_cb_scr_{idx}"))
        _set("shadows_mg", c2.slider("S:洋-绿", -100, 100, int(params.get("shadows_mg", 0)),
             key=f"p_cb_smg_{idx}"))
        _set("shadows_yb", c3.slider("S:黄-蓝", -100, 100, int(params.get("shadows_yb", 0)),
             key=f"p_cb_syb_{idx}"))
        st.markdown("**中间调**")
        c1, c2, c3 = st.columns(3)
        _set("midtones_cr", c1.slider("M:青-红", -100, 100, int(params.get("midtones_cr", 0)),
             key=f"p_cb_mcr_{idx}"))
        _set("midtones_mg", c2.slider("M:洋-绿", -100, 100, int(params.get("midtones_mg", 0)),
             key=f"p_cb_mmg_{idx}"))
        _set("midtones_yb", c3.slider("M:黄-蓝", -100, 100, int(params.get("midtones_yb", 0)),
             key=f"p_cb_myb_{idx}"))
        st.markdown("**高光**")
        c1, c2, c3 = st.columns(3)
        _set("highlights_cr", c1.slider("H:青-红", -100, 100, int(params.get("highlights_cr", 0)),
             key=f"p_cb_hcr_{idx}"))
        _set("highlights_mg", c2.slider("H:洋-绿", -100, 100, int(params.get("highlights_mg", 0)),
             key=f"p_cb_hmg_{idx}"))
        _set("highlights_yb", c3.slider("H:黄-蓝", -100, 100, int(params.get("highlights_yb", 0)),
             key=f"p_cb_hyb_{idx}"))

    elif t == "lut_filter":
        luts = list_available_luts()
        lut_items = list(luts.items())
        if lut_items:
            names = [v for _, v in lut_items]
            keys = [k for k, _ in lut_items]
            cur = params.get("preset_name", "vintage")
            sel = st.selectbox("滤镜预设", names, index=keys.index(cur) if cur in keys else 0,
                              key=f"p_lut_{idx}")
            _set("preset_name", keys[names.index(sel)])

    elif t == "curves":
        _set("brightness", st.slider("亮度", -100, 100, int(params.get("brightness", 0)),
             key=f"p_cv_b_{idx}"))
        _set("contrast", st.slider("对比度", -100, 100, int(params.get("contrast", 0)),
             key=f"p_cv_c_{idx}"))

    elif t == "resize_scale":
        _set("scale", st.slider("缩放比例", 0.05, 4.0, float(params.get("scale", 1.0)), 0.05,
             format="%.2f×", key=f"p_rs_{idx}"))

    elif t == "resize_fit":
        _set("width", st.number_input("最大宽度 (px)", 100, 20000,
             int(params.get("width", 1920)), key=f"p_rf_w_{idx}"))
        _set("height", st.number_input("最大高度 (px)", 100, 20000,
             int(params.get("height", 1080)), key=f"p_rf_h_{idx}"))

    elif t == "resize_long_edge":
        _set("max_long_edge", st.number_input("长边最大像素", 100, 50000,
             int(params.get("max_long_edge", 2048)), key=f"p_rle_{idx}"))

    elif t == "resize_short_edge":
        _set("min_short_edge", st.number_input("短边最小像素", 100, 20000,
             int(params.get("min_short_edge", 1024)), key=f"p_rse_{idx}"))

    elif t == "resize_exact":
        _set("width", st.number_input("目标宽度", 10, 50000,
             int(params.get("width", 800)), key=f"p_re_w_{idx}"))
        _set("height", st.number_input("目标高度", 10, 50000,
             int(params.get("height", 600)), key=f"p_re_h_{idx}"))
        _set("allow_upscale", st.checkbox("允许放大", params.get("allow_upscale", True),
             key=f"p_re_u_{idx}"))

    elif t == "smart_pad":
        _set("width", st.number_input("目标宽度", 100, 20000,
             int(params.get("width", 1080)), key=f"p_sp_w_{idx}"))
        _set("height", st.number_input("目标高度", 100, 20000,
             int(params.get("height", 1080)), key=f"p_sp_h_{idx}"))
        modes = ["auto", "color", "blur", "mirror", "repeat", "dominant"]
        labels = ["自动", "纯色", "模糊延伸", "镜像", "重复平铺", "主色调"]
        sel = st.selectbox("填充模式", labels, index=modes.index(params.get("mode", "auto")),
                          key=f"p_sp_m_{idx}")
        _set("mode", modes[labels.index(sel)])
        _set("bg_color", st.color_picker("背景色(纯色模式)", params.get("bg_color", "#ffffff"),
             key=f"p_sp_c_{idx}"))

    elif t == "uniform_resolution":
        _set("target_mp", st.slider("目标百万像素数 (MP)", 0.5, 100.0,
             float(params.get("target_mp", 24.0)), 0.5, format="%.1f MP", key=f"p_ur_{idx}"))

    elif t == "add_border":
        _set("border_width", st.number_input("边框宽度 (px)", 1, 500,
             int(params.get("border_width", 20)), key=f"p_ab_w_{idx}"))
        _set("color", st.color_picker("边框颜色", params.get("color", "#ffffff"),
             key=f"p_ab_c_{idx}"))

    elif t == "remove_bg":
        methods = ["auto", "ai", "traditional"]
        labels = ["自动(推荐)", "AI(rembg深度学习)", "传统(显著性+GrabCut)"]
        sel = st.selectbox("去背景方法", labels, index=methods.index(params.get("method", "auto")),
                          key=f"p_rb_m_{idx}")
        _set("method", methods[labels.index(sel)])
        rep = st.checkbox("背景替换为纯色(否则透明)", params.get("replace_with_color", False),
                          key=f"p_rb_r_{idx}")
        _set("replace_with_color", rep)
        if rep:
            _set("bg_color", st.color_picker("背景色", params.get("bg_color", "#ffffff"),
                 key=f"p_rb_c_{idx}"))

    elif t == "smart_sharpen":
        auto = st.checkbox("自动检测模糊并调整强度", params.get("auto_strength", True),
                          key=f"p_ss_a_{idx}")
        _set("auto_strength", auto)
        if not auto:
            _set("strength", st.slider("锐化强度", 0.1, 3.0,
                 float(params.get("strength", 1.0)), 0.1, key=f"p_ss_s_{idx}"))

    elif t == "unsharp_mask":
        _set("amount", st.slider("数量/强度", 0.1, 5.0, float(params.get("amount", 1.0)), 0.1,
             key=f"p_um_a_{idx}"))
        _set("radius", st.slider("半径", 0.1, 5.0, float(params.get("radius", 1.5)), 0.1,
             key=f"p_um_r_{idx}"))
        _set("threshold", st.slider("阈值", 0, 20, int(params.get("threshold", 2)),
             key=f"p_um_t_{idx}"))

    elif t == "restore_photo":
        _set("remove_scratches", st.checkbox("检测并修复划痕", params.get("remove_scratches", True),
             key=f"p_rp_rs_{idx}"))
        _set("denoise", st.checkbox("降噪处理", params.get("denoise", True),
             key=f"p_rp_d_{idx}"))
        _set("color_correct", st.checkbox("褪色校正", params.get("color_correct", True),
             key=f"p_rp_cc_{idx}"))
        _set("enhance_contrast", st.checkbox("对比度增强", params.get("enhance_contrast", True),
             key=f"p_rp_ec_{idx}"))

    elif t == "style_transfer":
        styles = list_available_styles()
        style_items = list(styles.items())
        if style_items:
            labels = [v for _, v in style_items]
            keys = [k for k, _ in style_items]
            cur = params.get("style", "oil")
            sel = st.selectbox("选择风格", labels, index=keys.index(cur) if cur in keys else 0,
                              key=f"p_st_{idx}")
            _set("style", keys[labels.index(sel)])

    elif t == "format_convert":
        fmts = ["JPEG", "PNG", "WEBP", "TIFF", "BMP", "HEIF"]
        fmt_labels = ["JPEG (有损, 通用)", "PNG (无损, 透明)", "WebP (现代, 体积小)",
                      "TIFF (无损, 专业打印)", "BMP (无压缩)", "HEIF (苹果格式)"]
        cur = params.get("target_format", "JPEG")
        sel = st.selectbox("目标格式", fmt_labels, index=fmts.index(cur) if cur in fmts else 0,
                          key=f"p_fc_f_{idx}")
        _set("target_format", fmts[fmt_labels.index(sel)])
        _set("quality", st.slider("压缩质量", 1, 100, int(params.get("quality", 95)),
             key=f"p_fc_q_{idx}"))
        _set("optimize", st.checkbox("优化编码", params.get("optimize", True),
             key=f"p_fc_o_{idx}"))
        _set("keep_exif", st.checkbox("保留EXIF元数据", params.get("keep_exif", True),
             key=f"p_fc_e_{idx}"))

    else:
        st.caption("此步骤无特殊参数，使用默认设置")
        for k, v in defaults.items():
            if k not in params:
                params[k] = v

    if changed:
        st.session_state.pipeline.steps[idx].params = params
        _invalidate_preview_cache()
        st.rerun()


def _run_batch_processing_ui():
    """执行批量处理并在主区域显示实时进度"""
    if not st.session_state.pipeline or not st.session_state.image_paths:
        st.session_state.processing = False
        return

    output_dir = tempfile.mkdtemp(prefix="output_batch_")
    st.session_state.output_dir = output_dir
    st.session_state._zip_ready = False
    st.session_state._zip_path = None

    total = len(st.session_state.image_paths)

    st.header("⏳ 正在批量处理...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.info(f"准备处理 {total} 张图片...")

    c1, c2, c3, c4 = st.columns(4)
    ph_success = c1.empty()
    ph_fail = c2.empty()
    ph_time = c3.empty()
    ph_speed = c4.empty()

    ph_success.markdown('<div class="big-number">0</div><div class="metric-label">✅ 成功</div>', unsafe_allow_html=True)
    ph_fail.markdown('<div class="big-number">0</div><div class="metric-label">❌ 失败</div>', unsafe_allow_html=True)
    ph_time.markdown('<div class="big-number">0s</div><div class="metric-label">⏱️ 已用</div>', unsafe_allow_html=True)
    ph_speed.markdown('<div class="big-number">0/s</div><div class="metric-label">🚀 速度</div>', unsafe_allow_html=True)

    metrics = {"success": 0, "fail": 0, "start": time.time()}

    def progress_cb(done, total_count, latest_result):
        pct = done / total_count
        progress_bar.progress(pct)
        elapsed = time.time() - metrics["start"]
        if latest_result.success:
            metrics["success"] += 1
        else:
            metrics["fail"] += 1
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total_count - done) / speed if speed > 0 else 0
        status_text.info(
            f"**进度**: {done}/{total_count} ({pct*100:.1f}%) · "
            f"**速度**: {speed:.1f} 张/秒 · "
            f"**剩余**: ~{eta:.0f}s"
        )
        ph_success.markdown(
            f'<div class="big-number">{metrics["success"]}</div>'
            f'<div class="metric-label">✅ 成功</div>',
            unsafe_allow_html=True,
        )
        ph_fail.markdown(
            f'<div class="big-number">{metrics["fail"]}</div>'
            f'<div class="metric-label">❌ 失败</div>',
            unsafe_allow_html=True,
        )
        ph_time.markdown(
            f'<div class="big-number">{elapsed:.0f}s</div>'
            f'<div class="metric-label">⏱️ 已用</div>',
            unsafe_allow_html=True,
        )
        ph_speed.markdown(
            f'<div class="big-number">{speed:.1f}/s</div>'
            f'<div class="metric-label">🚀 速度</div>',
            unsafe_allow_html=True,
        )

    rename_template = st.session_state.rename_template if st.session_state.rename_enabled else None

    try:
        results = batch_execute_pipeline(
            source_paths=st.session_state.image_paths,
            output_dir=output_dir,
            pipeline=st.session_state.pipeline,
            rename_template=rename_template,
            num_workers=st.session_state.num_workers,
            progress_callback=progress_cb,
        )
    except Exception as e:
        st.error(f"批量处理异常: {e}")
        results = []

    st.session_state.last_results = results
    st.session_state.processing = False
    st.session_state.processing_done = True
    st.session_state._results_dir = output_dir
    st.session_state._results_start = metrics["start"]
    st.session_state._results_metrics = dict(metrics)
    st.session_state._zip_ready = False
    st.session_state._zip_path = None
    st.rerun()


def _show_results_ui():
    """显示批量处理结果界面"""
    results = st.session_state.last_results
    metrics = st.session_state.get("_results_metrics", {})
    start_time = st.session_state.get("_results_start", time.time())
    output_dir = st.session_state.get("_results_dir", "")

    st.header("✅ 处理完成")

    total_elapsed = time.time() - start_time
    success_count = metrics.get("success", sum(1 for r in results if r.success))
    fail_count = metrics.get("fail", sum(1 for r in results if not r.success))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ 成功", f"{success_count} 张")
    c2.metric("❌ 失败", f"{fail_count} 张")
    c3.metric("⏱️ 总耗时", f"{total_elapsed:.1f} 秒")
    c4.metric("🚀 平均速度", f"{len(results)/max(0.1, total_elapsed):.1f} 张/秒")

    st.divider()
    st.subheader("📦 下载结果")

    col_zip, col_stats = st.columns([1, 1])

    with col_zip:
        if output_dir and os.path.isdir(output_dir):
            zip_path = st.session_state.get("_zip_path")
            if not zip_path or not os.path.isfile(zip_path):
                with st.spinner("正在打包 ZIP..."):
                    zip_path = os.path.join(tempfile.gettempdir(),
                                            f"processed_{int(time.time())}.zip")
                    create_zip(output_dir, zip_path)
                    st.session_state._zip_path = zip_path
                    st.session_state._zip_ready = True

            zip_size = os.path.getsize(zip_path) / (1024 * 1024)
            with open(zip_path, "rb") as f:
                st.download_button(
                    f"⬇️ 下载全部 ZIP ({zip_size:.1f} MB)",
                    f,
                    file_name=f"processed_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
        else:
            st.warning("输出目录不存在")

    with col_stats:
        failed = [r for r in results if not r.success]
        if failed:
            with st.expander(f"❌ 失败项 ({len(failed)})", expanded=True):
                for r in failed[:20]:
                    st.markdown(f"- **{Path(r.source_path).name}**: {r.error_message[:200]}")
                if len(failed) > 20:
                    st.caption(f"... 还有 {len(failed)-20} 项")
        else:
            st.success("🎉 全部处理成功！")

        times = [r.processing_time for r in results if r.success and r.processing_time > 0]
        if times:
            wall_time = total_elapsed
            st.markdown(f"""
            📊 **耗时统计**:
            - 平均: {np.mean(times):.2f} s/张
            - 最快: {np.min(times):.2f} s
            - 最慢: {np.max(times):.2f} s
            - 累计处理: {sum(times):.1f} s (实际耗时: {wall_time:.1f} s)
            - 并行加速比: {sum(times) / max(0.1, wall_time):.1f}×
            """)

    st.divider()
    if st.button("🔄 开始新任务", use_container_width=True, type="secondary"):
        st.session_state.processing = False
        st.session_state.processing_done = False
        st.session_state.last_results = []
        st.session_state._zip_ready = False
        st.session_state._zip_path = None
        st.session_state._results_dir = None
        st.session_state._results_start = None
        st.session_state._results_metrics = {}
        st.rerun()


# ============================================================
# 初始化 session_state
# ============================================================

SESSION_DEFAULTS = {
    "images": [],
    "image_paths": [],
    "input_dir": None,
    "selected_image_idx": 0,
    "output_dir": None,
    "processing": False,
    "processing_done": False,
    "last_results": [],
    "show_comparison": True,
    "rename_template": "{stem}_{index:04d}",
    "rename_enabled": False,
    "num_workers": max(1, (os.cpu_count() or 4) - 1),
    "expanded_step": -1,
    "drag_order": None,
    "_zip_ready": False,
    "_zip_path": None,
    "_results_dir": None,
    "_results_start": None,
    "_results_metrics": {},
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "pipeline" not in st.session_state:
    if Pipeline is not None:
        st.session_state.pipeline = Pipeline(name="我的流水线", steps=[])
    else:
        st.session_state.pipeline = None

if "preview_cache" not in st.session_state:
    st.session_state.preview_cache = {}

if _PIPELINE_READY:
    try:
        ensure_builtin_presets()
    except Exception:
        pass


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.title("🖼️ 图片批处理")
    st.caption("Python + Pillow + OpenCV + Streamlit")

    if _IMPORT_ERRORS:
        with st.expander("⚠️ 部分模块加载异常", expanded=False):
            for err in _IMPORT_ERRORS:
                st.caption(f"- `{err}`")
            st.info("核心功能仍可用，部分高级功能可能受限。请安装缺失依赖后重启。")

    st.divider()
    st.subheader("📁 数据源")

    tab_upload, tab_local = st.tabs(["📦 上传ZIP", "💻 本地目录"])

    with tab_upload:
        zip_file = st.file_uploader("上传包含图片的 ZIP 文件", type=["zip"])
        if zip_file:
            if st.button("解压并加载", use_container_width=True, type="primary"):
                if list_images_in_dir is not None:
                    with st.spinner("解压中..."):
                        tmpdir = tempfile.mkdtemp(prefix="img_batch_")
                        zip_path = os.path.join(tmpdir, "upload.zip")
                        with open(zip_path, "wb") as f:
                            f.write(zip_file.getbuffer())
                        extract_dir = extract_zip(zip_path, os.path.join(tmpdir, "extracted"))
                        image_paths = list_images_in_dir(extract_dir)
                        st.session_state.input_dir = extract_dir
                        st.session_state.image_paths = image_paths
                        st.session_state.selected_image_idx = 0
                        _invalidate_preview_cache()
                        st.rerun()

    with tab_local:
        local_dir = st.text_input("本地目录路径", value=r"D:\photos", help="支持绝对路径")
        if st.button("扫描目录", use_container_width=True, type="primary"):
            if list_images_in_dir is not None and os.path.isdir(local_dir):
                image_paths = list_images_in_dir(local_dir)
                st.session_state.input_dir = local_dir
                st.session_state.image_paths = image_paths
                st.session_state.selected_image_idx = 0
                _invalidate_preview_cache()
                st.rerun()
            elif not os.path.isdir(local_dir):
                st.error("目录不存在")

    st.divider()

    if st.session_state.image_paths:
        total = len(st.session_state.image_paths)
        st.success(f"✅ 已加载 **{total}** 张图片")
        total_size = 0
        for p in st.session_state.image_paths[:20]:
            try:
                total_size += os.path.getsize(p)
            except Exception:
                pass
        avg = total_size / max(1, min(20, total)) / (1024 * 1024)
        est_total = avg * total
        st.caption(f"估计总大小: ~{est_total:.1f} MB (平均 {avg:.2f} MB/张)")

    st.divider()
    st.subheader("⚙️ 输出设置")
    st.session_state.num_workers = st.slider(
        "并行进程数", 1, max(1, os.cpu_count() or 4),
        st.session_state.num_workers,
        help=f"检测到 CPU {os.cpu_count() or '?'} 核心",
    )
    st.session_state.rename_enabled = st.checkbox(
        "启用批量重命名", value=st.session_state.rename_enabled)
    if st.session_state.rename_enabled:
        tpl = st.text_input(
            "命名模板",
            value=st.session_state.rename_template,
            help="{stem}原名 {index}序号 {date}拍摄日期 {gps_short}位置简写",
        )
        st.session_state.rename_template = tpl
        with st.expander("📖 模板变量参考"):
            st.markdown("""
            **基础**: `{stem}` `{ext}` `{index}` `{index04d}` `{total}`

            **日期**: `{date}` `{datetime}` `{year}` `{month}` `{day}` `{hour}` `{minute}` `{second}`

            **相机**: `{camera}` `{lens}` `{focal}` `{aperture}` `{shutter}` `{iso}`

            **尺寸**: `{width}` `{height}` `{size}`

            **位置 (GPS)**:
            - `{gps_lat}` 纬度 (如 39.9042)
            - `{gps_lon}` 经度 (如 116.4074)
            - `{gps_dms}` 度分秒 (如 39N5433_116E2426)
            - `{gps_short}` 简写 (如 40N116E)
            - `{location}` 区域 (如 SubTrop_EAsia)

            无 GPS 时显示 NoGPS / NoLoc
            """)


# ============================================================
# 主界面 - 根据状态显示不同内容
# ============================================================

is_processing = st.session_state.processing
is_done = st.session_state.processing_done

# --- 正在处理: 显示进度页 ---
if is_processing and not is_done:
    _run_batch_processing_ui()

# --- 处理完成: 显示结果页 ---
elif is_done:
    _show_results_ui()

# --- 正常模式: 显示图片库 + 流水线 ---
else:
    col_main, col_pipeline = st.columns([2, 1])

    # ============================================================
    # 主区域 - 图片库 + 预览
    # ============================================================
    with col_main:
        st.header("📷 图片库")

        if not st.session_state.image_paths:
            st.info("👈 请在左侧上传 ZIP 文件或指定本地目录来加载图片")
            st.markdown("""
            ---
            ### 功能概览

            | 类别 | 功能 |
            |------|------|
            | 🧠 **智能裁剪** | 人脸居中、主体检测、三分法构图、固定比例 |
            | 🎨 **颜色调整** | 自动白平衡、色调曲线、HSL、10+ LUT滤镜 |
            | 📐 **尺寸处理** | 缩放、分辨率统一、长边/短边限制、智能填充 |
            | 🔄 **格式转换** | JPEG/PNG/WebP/HEIF/TIFF、EXIF保留/剥离 |
            | 🏷️ **批量重命名** | EXIF日期、GPS位置、相机型号、自定义模板 |
            | 🤖 **AI 辅助** | 自动去背景、智能锐化、老照片修复、7种风格 |
            | 🏗️ **流水线** | 拖拽排序、实时预览、多进程并行、预设保存 |
            """)
        else:
            with st.expander(f"📸 图片缩略图 ({len(st.session_state.image_paths)} 张)", expanded=True):
                cols_per_row = 6
                total = len(st.session_state.image_paths)
                show_max = min(60, total)
                rows = (show_max + cols_per_row - 1) // cols_per_row

                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        idx = r * cols_per_row + c
                        if idx >= show_max:
                            break
                        with cols[c]:
                            try:
                                thumb = generate_thumbnail(
                                    st.session_state.image_paths[idx], (150, 150))
                                is_selected = idx == st.session_state.selected_image_idx
                                label = "✅ " if is_selected else ""
                                if st.button(
                                    f"{label}{idx+1}",
                                    key=f"thumb_{idx}",
                                    use_container_width=True,
                                    type="primary" if is_selected else "secondary",
                                ):
                                    st.session_state.selected_image_idx = idx
                                    _invalidate_preview_cache()
                                    st.rerun()
                                st.image(thumb, caption=None, use_container_width=True)
                            except Exception:
                                st.error("⚠️", icon=None)

                if total > show_max:
                    st.caption(f"... 还有 {total - show_max} 张未显示")

            st.divider()
            selected_idx = st.session_state.selected_image_idx
            if 0 <= selected_idx < len(st.session_state.image_paths):
                img_path = st.session_state.image_paths[selected_idx]

                try:
                    info = get_image_info(img_path) if get_image_info else None
                except Exception:
                    info = None

                col_prev_ctrls, col_prev_info = st.columns([3, 2])
                with col_prev_ctrls:
                    st.subheader(f"🖼️ {Path(img_path).name}")
                    st.session_state.show_comparison = st.checkbox(
                        "显示处理后对比", value=st.session_state.show_comparison,
                    )
                    prev_col, next_col = st.columns(2)
                    with prev_col:
                        if st.button("⬅️ 上一张", use_container_width=True,
                                    disabled=selected_idx == 0):
                            st.session_state.selected_image_idx -= 1
                            _invalidate_preview_cache()
                            st.rerun()
                    with next_col:
                        if st.button("下一张 ➡️", use_container_width=True,
                                    disabled=selected_idx >= len(st.session_state.image_paths) - 1):
                            st.session_state.selected_image_idx += 1
                            _invalidate_preview_cache()
                            st.rerun()

                with col_prev_info:
                    if info:
                        st.markdown(f"""
                        **📊 图像信息**
                        - 📐 尺寸: **{info.width} × {info.height}** ({(info.width*info.height)/1e6:.1f} MP)
                        - 🎨 格式: `{info.format}` / `{info.mode}`
                        - 💾 文件: **{info.size_mb:.2f} MB**
                        """)
                        exif_items = [
                            ("相机", info.exif.get("0th.Model", "")),
                            ("拍摄时间", str(info.exif.get("Exif.DateTimeOriginal", ""))),
                            ("焦距", str(info.exif.get("Exif.FocalLength", ""))),
                            ("光圈", str(info.exif.get("Exif.FNumber", ""))),
                            ("ISO", str(info.exif.get("Exif.ISOSpeedRatings", ""))),
                        ]
                        exif_str = "\n".join([
                            f"- **{k}**: `{str(v)[:30]}`" for k, v in exif_items if v
                        ])
                        if exif_str:
                            st.markdown(f"**📷 EXIF**\n{exif_str}")

                try:
                    original = load_image(img_path) if load_image else None
                except Exception as e:
                    st.error(f"加载图片失败: {e}")
                    original = None

                if original is not None and st.session_state.pipeline is not None:
                    processed = None
                    step_results = []
                    if st.session_state.pipeline.steps:
                        cache_key = (img_path, json.dumps(
                            st.session_state.pipeline.to_dict(), sort_keys=True))
                        if cache_key in st.session_state.preview_cache:
                            processed, step_results = st.session_state.preview_cache[cache_key]
                        else:
                            with st.spinner("处理预览中..."):
                                processed, step_results = execute_pipeline(
                                    original,
                                    st.session_state.pipeline,
                                    capture_intermediate=True,
                                )
                                st.session_state.preview_cache[cache_key] = (processed, step_results)

                    if st.session_state.show_comparison and processed is not None:
                        col_orig, col_proc = st.columns(2)
                        with col_orig:
                            st.markdown("**📷 原图**")
                            st.image(original, use_container_width=True)
                        with col_proc:
                            st.markdown(f"**✨ 处理后 ({len(st.session_state.pipeline.steps)} 步骤)**")
                            st.image(processed, use_container_width=True)

                            buf = io.BytesIO()
                            try:
                                ext = getattr(processed, "_format_hint", "JPEG")
                                if ext in ("HEIC", "HEIF"):
                                    ext = "PNG"
                                processed.save(buf, format=ext, quality=90)
                                st.download_button(
                                    "⬇️ 下载预览", buf.getvalue(),
                                    file_name=f"processed_{selected_idx:04d}.{ext.lower()}",
                                    mime=f"image/{ext.lower()}",
                                    use_container_width=True,
                                )
                            except Exception:
                                pass
                    elif original is not None:
                        st.image(original, use_container_width=True)

                    if processed is not None and st.session_state.pipeline.steps:
                        with st.expander("🔍 每一步效果预览", expanded=False):
                            cache_key = (img_path, json.dumps(
                                st.session_state.pipeline.to_dict(), sort_keys=True))
                            _, step_results_s = st.session_state.preview_cache.get(
                                cache_key, (None, []))
                            enabled_steps = [s for s in step_results_s
                                            if s.get("enabled") and "image" in s]
                            if enabled_steps:
                                step_cols = st.columns(min(5, len(enabled_steps)))
                                for i, (sc, sr) in enumerate(zip(step_cols, enabled_steps)):
                                    with sc:
                                        info_step = STEP_REGISTRY.get(
                                            sr["step_type"], {})
                                        st.caption(f"**{i+1}. {info_step.get('name', sr['step_type'])}**")
                                        try:
                                            st.image(sr["image"], use_container_width=True)
                                        except Exception:
                                            pass
                                        st.caption(f"⏱️ {sr.get('time_ms', 0):.0f}ms")

    # ============================================================
    # 右侧 - 流水线面板
    # ============================================================
    with col_pipeline:
        st.header("🏗️ 处理流水线")

        if not _PIPELINE_READY:
            st.error("⚠️ 流水线引擎未能加载，请检查依赖安装")
            st.stop()

        preset_tab_add, preset_tab_list = st.tabs(["💾 保存预设", "📂 加载预设"])
        with preset_tab_add:
            preset_name = st.text_input("预设名称", placeholder="我的自定义预设")
            preset_desc = st.text_area("预设描述", placeholder="简单描述此流水线...", height=60)
            if st.button("保存当前流水线为预设", use_container_width=True):
                if preset_name and st.session_state.pipeline:
                    save_preset(preset_name, st.session_state.pipeline, preset_desc)
                    st.success(f"已保存预设: {preset_name}")

        with preset_tab_list:
            presets = list_presets()
            if not presets:
                st.info("暂无预设")
            else:
                for p in presets:
                    with st.container():
                        col_pn, col_lo, col_de = st.columns([4, 1, 1])
                        with col_pn:
                            st.markdown(f"**{p['name']}**")
                            st.caption(f"{p['step_count']} 步 · {p['description'][:40]}")
                        with col_lo:
                            if st.button("加载", key=f"load_{p['name']}", type="primary"):
                                loaded = load_preset(p["name"])
                                if loaded:
                                    st.session_state.pipeline = loaded["pipeline"]
                                    st.session_state.drag_order = None
                                    st.session_state.expanded_step = -1
                                    _invalidate_preview_cache()
                                    st.rerun()
                        with col_de:
                            if st.button("🗑️", key=f"del_{p['name']}", help="删除预设"):
                                delete_preset(p["name"])
                                st.rerun()
                        st.divider()

        with st.expander("➕ 添加处理步骤", expanded=True):
            categories = get_step_categories()
            for cat, steps in categories.items():
                st.markdown(f"**{cat}**")
                for step in steps:
                    if st.button(
                        f"＋ {step['name']}",
                        key=f"add_{step['step_type']}",
                        use_container_width=True,
                        help=step["description"],
                    ):
                        new_step = PipelineStep(
                            step_type=step["step_type"],
                            enabled=True,
                            params=dict(step["default_params"]),
                        )
                        st.session_state.pipeline.steps.append(new_step)
                        st.session_state.drag_order = None
                        st.session_state.expanded_step = len(st.session_state.pipeline.steps) - 1
                        _invalidate_preview_cache()
                        st.rerun()

        st.divider()
        st.subheader(f"📋 步骤列表 ({len(st.session_state.pipeline.steps)})")

        if not st.session_state.pipeline.steps:
            st.info("👆 从上方添加处理步骤，或加载预设")
        else:
            n_steps = len(st.session_state.pipeline.steps)

            # --- 拖拽排序 ---
            if _HAS_SORTABLES and n_steps > 1:
                st.markdown("**🔀 拖拽调整顺序**")
                drag_items = []
                for i, step in enumerate(st.session_state.pipeline.steps):
                    si = _get_step_info(step.step_type)
                    drag_items.append(f"{i+1}. [{si.get('category','')}] {si['name']}")
                new_order = sorted_items(drag_items, multi_containers=False, key="sortable_steps")
                if new_order and len(new_order) == n_steps:
                    old_keys = []
                    for j in range(n_steps):
                        si = _get_step_info(st.session_state.pipeline.steps[j].step_type)
                        old_keys.append(f"{j+1}. [{si.get('category','')}] {si['name']}")
                    if new_order != old_keys:
                        idx_map = {old_keys[j]: j for j in range(n_steps)}
                        reordered = [st.session_state.pipeline.steps[idx_map[item]]
                                    for item in new_order if item in idx_map]
                        if len(reordered) == n_steps:
                            st.session_state.pipeline.steps = reordered
                            st.session_state.expanded_step = -1
                            _invalidate_preview_cache()
                            st.rerun()
            elif n_steps > 1:
                st.markdown("**🔀 排序（输入位置编号）**")
                cols_drag = st.columns(min(n_steps, 8))
                drag_positions = []
                for i in range(min(n_steps, 8)):
                    with cols_drag[i]:
                        si = _get_step_info(st.session_state.pipeline.steps[i].step_type)
                        pos = st.number_input(
                            f"{i+1}.{si['name'][:8]}",
                            min_value=1, max_value=n_steps,
                            value=i + 1, step=1,
                            key=f"drag_{i}",
                            label_visibility="collapsed",
                        )
                        drag_positions.append(pos)

                if len(drag_positions) == n_steps and n_steps > 1:
                    new_order = sorted(range(n_steps), key=lambda x: drag_positions[x])
                    reordered = [st.session_state.pipeline.steps[new_order[i]]
                                for i in range(n_steps)]
                    old_serialized = json.dumps(
                        [s.to_dict() for s in st.session_state.pipeline.steps], sort_keys=True)
                    new_serialized = json.dumps(
                        [s.to_dict() for s in reordered], sort_keys=True)
                    if old_serialized != new_serialized:
                        st.session_state.pipeline.steps = reordered
                        st.session_state.expanded_step = -1
                        _invalidate_preview_cache()
                        st.rerun()

            st.divider()

            # --- 步骤卡片列表 ---
            for i, step in enumerate(st.session_state.pipeline.steps):
                step_info = _get_step_info(step.step_type)
                cat = step_info.get("category", "")
                tag_class = {
                    "裁剪": "tag-crop", "颜色": "tag-color", "尺寸": "tag-size",
                    "AI": "tag-ai", "输出": "tag-output",
                }.get(cat, "")

                enabled_class = "enabled" if step.enabled else "disabled"
                is_expanded = (st.session_state.expanded_step == i)

                # 步骤标题行
                head_col1, head_col2 = st.columns([5, 1])
                with head_col1:
                    st.markdown(f"""
                    <div class="step-card {enabled_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="tag {tag_class}">{cat}</span>
                                <strong>{i+1}. {step_info['name']}</strong>
                            </div>
                            <small style="opacity:0.7">{step.step_type}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with head_col2:
                    # 展开/折叠按钮
                    btn_label = "▼" if is_expanded else "▶"
                    if st.button(btn_label, key=f"toggle_{i}", use_container_width=True,
                                help="展开/折叠参数"):
                        if is_expanded:
                            st.session_state.expanded_step = -1
                        else:
                            st.session_state.expanded_step = i
                        st.rerun()

                # 操作按钮行
                col_on, col_up, col_dn, col_rm = st.columns([1, 1, 1, 1])
                with col_on:
                    new_enabled = st.checkbox(
                        "", value=step.enabled, key=f"en_{i}",
                        label_visibility="collapsed",
                    )
                    if new_enabled != step.enabled:
                        st.session_state.pipeline.steps[i].enabled = new_enabled
                        _invalidate_preview_cache()
                        st.rerun()
                with col_up:
                    if st.button("⬆", key=f"up_{i}", disabled=(i == 0), help="上移"):
                        s = st.session_state.pipeline.steps.pop(i)
                        st.session_state.pipeline.steps.insert(i - 1, s)
                        if st.session_state.expanded_step == i:
                            st.session_state.expanded_step = i - 1
                        elif st.session_state.expanded_step == i - 1:
                            st.session_state.expanded_step = i
                        _invalidate_preview_cache()
                        st.rerun()
                with col_dn:
                    if st.button("⬇", key=f"dn_{i}",
                                disabled=(i == len(st.session_state.pipeline.steps) - 1),
                                help="下移"):
                        s = st.session_state.pipeline.steps.pop(i)
                        st.session_state.pipeline.steps.insert(i + 1, s)
                        if st.session_state.expanded_step == i:
                            st.session_state.expanded_step = i + 1
                        elif st.session_state.expanded_step == i + 1:
                            st.session_state.expanded_step = i
                        _invalidate_preview_cache()
                        st.rerun()
                with col_rm:
                    if st.button("🗑", key=f"rm_{i}", help="删除"):
                        st.session_state.pipeline.steps.pop(i)
                        if st.session_state.expanded_step >= i:
                            st.session_state.expanded_step = max(-1, st.session_state.expanded_step - 1)
                        _invalidate_preview_cache()
                        st.rerun()

                # 参数面板（独立控制展开/折叠）
                if is_expanded and step.enabled:
                    with st.container():
                        st.caption("**⚙️ 参数配置**")
                        _render_step_params(i, step, step_info["default_params"])
                elif not step.enabled:
                    st.caption("此步骤已禁用")

                st.caption("")  # 间距

        st.divider()

        # --- 开始批量处理按钮 ---
        has_images = bool(st.session_state.image_paths)
        has_steps = bool(st.session_state.pipeline and st.session_state.pipeline.steps)
        is_proc = st.session_state.processing
        is_done_btn = st.session_state.processing_done

        can_start = has_images and has_steps and not is_proc and not is_done_btn

        if not has_images:
            st.caption("⚠️ 请先加载图片")
        elif not has_steps:
            st.caption("⚠️ 请先添加处理步骤")

        if st.button(
            "🚀 开始批量处理",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
        ):
            st.session_state.processing = True
            st.session_state.processing_done = False
            st.rerun()
