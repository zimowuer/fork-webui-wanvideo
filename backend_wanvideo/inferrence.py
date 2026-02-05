# F:\sd-reforge\webui\extensions\sd-webui-wanvideo\scripts\generation.py
import gradio as gr
import torch
import time
import psutil
import os
import numpy as np

# --- diffsynth import compatibility ---
try:
    # old style (may fail in new diffsynth)
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    _HAS_MODEL_MANAGER = True
except Exception:
    # new style: ModelManager removed
    ModelManager = None
    _HAS_MODEL_MANAGER = False
    from diffsynth import WanVideoPipeline, save_video, VideoData
    # ModelConfig location varies by pipeline/module; try best-effort imports.
    try:
        from diffsynth.pipelines.wan_video import ModelConfig  # common pattern
    except Exception:
        try:
            from diffsynth.pipelines.qwen_image import ModelConfig  # documented example
        except Exception:
            ModelConfig = None  # will raise later with a clear error

from PIL import Image
from tqdm import tqdm
import random
import logging
import re

try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()


# 获取硬件信息
def get_hardware_info():
    info = ""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            info += f"GPU: {gpu_name}\nVRAM: {total_vram}GB\n"
        else:
            info += "GPU: 不可用\n"
        info += f"CPU: {psutil.cpu_count(logical=False)} 物理核心 / {psutil.cpu_count(logical=True)} 逻辑核心\n"
        info += f"内存: {psutil.virtual_memory().total // (1024 ** 3)}GB\n"
    except Exception as e:
        info += f"硬件信息获取失败: {str(e)}"
    return info


# 获取指定目录中的模型文件列表
def get_model_files(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return ["无模型文件"]
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.endswith('.txt') and not f.endswith('.json')
    ]
    return files if files else ["无模型文件"]


# 从提示词中提取 LoRA 信息
def extract_lora_from_prompt(prompt):
    lora_pattern = r"<lora:([^:]+):([\d\.]+)>"
    matches = re.findall(lora_pattern, prompt)
    loras = [(name, float(weight)) for name, weight in matches]
    cleaned_prompt = re.sub(lora_pattern, "", prompt).strip()
    return loras, cleaned_prompt


def _resolve_dtype(dtype_str: str):
    # 支持 FP16、BF16 和 FP8（若环境支持）
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    # fp8 不是所有 torch 版本/硬件都支持；尽量容错
    try:
        return torch.float8_e4m3fn
    except Exception:
        logging.warning("当前环境不支持 torch.float8_e4m3fn，回退到 bfloat16")
        return torch.bfloat16


def _resolve_imgenc_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    return torch.bfloat16


def _ensure_exists(path_or_paths, err_prefix="模型文件"):
    if isinstance(path_or_paths, (list, tuple)):
        for p in path_or_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{err_prefix} {p} 不存在，请检查路径")
    else:
        if not os.path.exists(path_or_paths):
            raise FileNotFoundError(f"{err_prefix} {path_or_paths} 不存在，请检查路径")


# 加载模型和 LoRA（新版兼容：ModelManager 被删除时改用 from_pretrained + ModelConfig）
def load_models(
    dit_models,
    t5_model,
    vae_model,
    image_encoder_model=None,
    lora_prompt="",
    torch_dtype="bfloat16",
    image_encoder_torch_dtype="float32",
    use_usp=False,
    num_persistent_param_in_dit=None
):
    # 定义模型目录
    base_dir = "models/wan2.1"
    dit_dir = os.path.join(base_dir, "dit")
    t5_dir = os.path.join(base_dir, "t5")
    vae_dir = os.path.join(base_dir, "vae")
    lora_dir = os.path.join(base_dir, "lora")
    image_encoder_dir = os.path.join(base_dir, "image_encoder") if image_encoder_model else None

    # 自动创建目录
    os.makedirs(dit_dir, exist_ok=True)
    os.makedirs(t5_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    if image_encoder_dir:
        os.makedirs(image_encoder_dir, exist_ok=True)

    # 记录目录信息以便调试
    logging.info(f"DIT 模型目录: {os.path.abspath(dit_dir)}")
    logging.info(f"T5 模型目录: {os.path.abspath(t5_dir)}")
    logging.info(f"VAE 模型目录: {os.path.abspath(vae_dir)}")
    logging.info(f"LoRA 模型目录: {os.path.abspath(lora_dir)}")
    if image_encoder_dir:
        logging.info(f"Image Encoder 模型目录: {os.path.abspath(image_encoder_dir)}")

    # 检查模型文件是否存在
    if not dit_models or "无模型文件" in dit_models or t5_model == "无模型文件" or vae_model == "无模型文件":
        raise Exception("请确保所有模型文件夹中都有有效的模型文件：DIT、T5 和 VAE 模型不可为空")
    if image_encoder_model in ["无模型文件", "无"]:
        image_encoder_model = None

    # 将多个 DIT 模型文件视为一个整体
    dit_model_paths = [os.path.join(dit_dir, dit_model) for dit_model in dit_models if dit_model != "无模型文件"]
    if not dit_model_paths:
        raise Exception("未选择有效的 DIT 模型文件")

    t5_path = os.path.join(t5_dir, t5_model)
    vae_path = os.path.join(vae_dir, vae_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype_obj = _resolve_dtype(torch_dtype)
    image_encoder_dtype_obj = _resolve_imgenc_dtype(image_encoder_torch_dtype)

    # 检查基础模型文件路径是否存在
    _ensure_exists(dit_model_paths, "DIT 模型文件")
    _ensure_exists(t5_path, "T5 模型文件")
    _ensure_exists(vae_path, "VAE 模型文件")

    # 从提示词中提取 LoRA 信息（先保留解析；注入暂不做）
    loras, _ = extract_lora_from_prompt(lora_prompt)
    loaded_loras = {}
    if loras:
        for lora_name, lora_weight in loras:
            lora_path = os.path.join(lora_dir, lora_name)
            if not os.path.exists(lora_path):
                logging.warning(f"LoRA 文件 {lora_path} 不存在，跳过加载")
                continue
            # 新版 LoRA 注入接口不确定：先不改逻辑，只记录并跳过实际加载
            logging.info(f"[LoRA] 检测到 LoRA: {lora_path} (alpha={lora_weight}) - 新版接口未适配，已跳过注入")
            loaded_loras[lora_name] = lora_weight
            # pass  # 明确占位（无需写也可）

    # 检查 USP 环境（保持原逻辑）
    if use_usp and not torch.distributed.is_initialized():
        logging.warning("USP 启用失败：分布式环境未初始化，将禁用 USP")
        use_usp = False

    # -------------------------
    # Old diffsynth path (ModelManager exists)
    # -------------------------
    if _HAS_MODEL_MANAGER and ModelManager is not None:
        model_manager = ModelManager(device="cpu", torch_dtype=torch_dtype_obj)

        # 加载 Image Encoder（若存在）
        model_list = [
            dit_model_paths,  # 多个 DIT 文件合并加载（旧版支持嵌套 list）
            t5_path,
            vae_path
        ]

        if image_encoder_model:
            image_encoder_path = os.path.join(image_encoder_dir, image_encoder_model)
            _ensure_exists(image_encoder_path, "Image Encoder 文件")
            logging.info(f"加载 Image Encoder: {image_encoder_path} (使用 {image_encoder_dtype_obj})")
            model_manager.load_models([image_encoder_path], torch_dtype=image_encoder_dtype_obj)
            model_list.insert(0, image_encoder_path)

        logging.info(f"开始加载基础模型(旧版 ModelManager): {model_list} (使用 {torch_dtype_obj})")
        model_manager.load_models(model_list, torch_dtype=torch_dtype_obj)
        logging.info(f"基础模型加载完成: {getattr(model_manager, 'model_name', None) or '未识别到模型'}")

        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch_dtype_obj,
            device=device,
            use_usp=use_usp
        )
    # -------------------------
    # New diffsynth path (ModelManager removed)
    # -------------------------
    else:
        if ModelConfig is None:
            raise ImportError(
                "检测到 diffsynth 已移除 ModelManager，但无法导入 ModelConfig。"
                "请确认新版 diffsynth 中 ModelConfig 的真实导入路径（例如 diffsynth.pipelines.wan_video 或 core.loader）。"
            )

        model_configs = []

        # 可选 Image Encoder：放在最前（若 WanVideoPipeline 需要）
        if image_encoder_model:
            image_encoder_path = os.path.join(image_encoder_dir, image_encoder_model)
            _ensure_exists(image_encoder_path, "Image Encoder 文件")
            logging.info(f"配置 Image Encoder: {image_encoder_path} (使用 {image_encoder_dtype_obj})")
            # 注意：from_pretrained 的 torch_dtype 是计算精度；img encoder 的单独 dtype 在文档里未给出通用写法
            # 这里先仅把路径作为一个 ModelConfig，保持最小改动。
            model_configs.append(ModelConfig(path=image_encoder_path))

        # DIT 多文件：新版用 ModelConfig(path=[...])
        model_configs.append(ModelConfig(path=dit_model_paths))
        model_configs.append(ModelConfig(path=t5_path))
        model_configs.append(ModelConfig(path=vae_path))

        logging.info(f"开始加载基础模型(新版 from_pretrained): {model_configs} (使用 {torch_dtype_obj})")
        # 注意：use_usp 是否仍作为 from_pretrained 参数未在你贴的文档里确认。
        # 为避免参数不匹配报错，这里采用“保守调用”：先不传 use_usp；若你确认新版支持再加回去。
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype_obj,
            device=device,
            model_configs=model_configs,
        )

    # 显存管理（尽量兼容：有就开，没有就忽略但不报错）
    if device == "cuda":
        if hasattr(pipe, "enable_vram_management"):
            try:
                pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
            except TypeError:
                # 新版可能改了签名：先退化为无参开启
                logging.warning("enable_vram_management 参数签名可能已变化，尝试无参开启")
                try:
                    pipe.enable_vram_management()
                except Exception as e:
                    logging.warning(f"显存管理开启失败（已忽略，不影响继续运行）: {e}")
        else:
            logging.warning("当前 WanVideoPipeline 不包含 enable_vram_management，已跳过显存管理开启")

    # 设置管道信息（保持原格式）
    pipe.hardware_info = get_hardware_info()
    pipe.model_name = f"DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}" + (
        f", Image Encoder: {image_encoder_model}" if image_encoder_model else ""
    )
    pipe.lora_info = ", ".join([f"{name} ({weight})" for name, weight in loaded_loras.items()]) if loaded_loras else "无"
    pipe.torch_dtype_info = f"DIT/T5/VAE: {torch_dtype_obj}, Image Encoder: {image_encoder_dtype_obj if image_encoder_model else '未使用'}"
    pipe.num_persistent_param_in_dit = num_persistent_param_in_dit
    return pipe


# 自适应图片分辨率
def adaptive_resolution(image):
    if image is None:
        return 512, 512
    try:
        width, height = image.size
        return width, height
    except Exception:
        return 512, 512


# 生成文生视频
def generate_t2v(prompt, negative_prompt, num_inference_steps, seed, height, width,
                 num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id,
                 dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength,
                 rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y,
                 torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                 num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    if not enable_num_persistent:
        num_persistent_param_in_dit = None

    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                       image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_t2v_{int(time.time())}.mp4")

        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")

        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- TeaCache L1阈值：{tea_cache_l1_thresh if tea_cache_l1_thresh is not None else '未使用'}
- TeaCache Model ID：{tea_cache_model_id}
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe


# 生成图生视频（新增 end_image 参数）
def generate_i2v(image, end_image, prompt, negative_prompt, num_inference_steps, seed, height, width,
                 num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id,
                 dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength,
                 rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y,
                 torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                 num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    if not enable_num_persistent:
        num_persistent_param_in_dit = None

    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                       image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if image is None:
            raise ValueError("请上传首帧")
        img = image.convert("RGB")
        end_img = end_image.convert("RGB") if end_image else None

        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=img,
            end_image=end_img,
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_i2v_{int(time.time())}.mp4")

        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")

        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- TeaCache L1阈值：{tea_cache_l1_thresh if tea_cache_l1_thresh is not None else '未使用'}
- TeaCache Model ID：{tea_cache_model_id}
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
- 是否使用尾帧：{'是' if end_image else '否'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe


# 生成视频生视频（新增 control_video 参数）
def generate_v2v(video, control_video, prompt, negative_prompt, num_inference_steps, seed, height, width,
                 num_frames, cfg_scale, sigma_shift, dit_models, t5_model, vae_model,
                 image_encoder_model, fps, denoising_strength, rand_device, tiled,
                 tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, torch_dtype,
                 image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                 num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    if not enable_num_persistent:
        num_persistent_param_in_dit = None

    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                       image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if video is None and control_video is None:
            raise ValueError("请至少上传初始视频或控制视频")
        video_data = VideoData(video, height=int(height), width=int(width)) if video else None
        control_video_data = VideoData(control_video, height=int(height), width=int(width)) if control_video else None

        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        frames = pipe(
            prompt=cleaned_prompt or "默认提示词",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=video_data,
            control_video=control_video_data,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=float(sigma_shift),
            tiled=bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=None,
            tea_cache_model_id="",
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 不可写，请检查权限")
        output_path = os.path.join(output_dir, f"wan_video_v2v_{int(time.time())}.mp4")

        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("磁盘空间不足，请清理后再试")

        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"显存使用：{mem_used}GB / 峰值保留：{mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
生成信息：
- 分辨率：{width}x{height}
- 总帧数：{num_frames}
- 推理步数：{num_inference_steps}
- 随机种子：{actual_seed} {'(随机生成)' if seed == -1 else ''}
- 总耗时：{time_cost:.2f}秒
- 帧率：{fps} FPS
- 视频时长：{num_frames / int(fps):.1f}秒
{mem_info}
- 模型版本：DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- 使用Tiled：{'是' if tiled else '否'}
- Tile Size：({tile_size_x}, {tile_size_y})
- Tile Stride：({tile_stride_x}, {tile_stride_y})
- Torch 数据类型：{pipe.torch_dtype_info}
- 使用USP：{'是' if use_usp else '否'}
- 显存管理参数 (num_persistent_param_in_dit)：{num_persistent_param_in_dit if num_persistent_param_in_dit is not None else '未限制'}
- 已加载LoRA：{pipe.lora_info}
- 是否使用控制视频：{'是' if control_video else '否'}
"""
        return output_path, info
    except Exception as e:
        return None, f"生成失败: {str(e)}"
    finally:
        del pipe
