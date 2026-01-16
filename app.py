"""
Video Style Editor - 使用Qwen Image Edit API进行视频风格编辑
"""

import os
import base64
import mimetypes
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import random
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 默认输出目录
DEFAULT_OUTPUT_DIR = Path("./output")
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"


def extract_frames(video_path: str, interval: float, output_dir: Path) -> tuple[list[str], float, tuple[int, int]]:
    """
    从视频中按指定时间间隔提取帧

    Args:
        video_path: 视频文件路径
        interval: 提取间隔（秒）
        output_dir: 输出目录

    Returns:
        (帧文件路径列表, fps, (宽度, 高度))
    """
    logger.info(f"开始提取帧: {video_path}, 间隔: {interval}秒")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info(f"视频信息: {width}x{height}, {fps:.2f}fps, 总帧数: {total_frames}, 时长: {duration:.2f}秒")

    # 计算要提取的帧
    frame_interval = int(fps * interval)
    if frame_interval < 1:
        frame_interval = 1

    frames_dir = output_dir / "original_frames"
    frames_dir.mkdir(exist_ok=True)

    frame_paths = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_path = frames_dir / f"frame_{extracted_count:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            extracted_count += 1

        frame_idx += 1

    cap.release()
    logger.info(f"帧提取完成: 共提取 {extracted_count} 帧")

    return frame_paths, fps, (width, height)


def extract_audio(video_path: str, output_dir: Path) -> str | None:
    """从视频中提取音频"""
    logger.info(f"开始提取音频: {video_path}")
    try:
        from moviepy import VideoFileClip

        audio_path = output_dir / "audio.mp3"
        video = VideoFileClip(video_path)

        if video.audio is not None:
            logger.info("检测到音频轨道，正在提取...")
            video.audio.write_audiofile(str(audio_path), logger=None)
            video.close()
            logger.info(f"音频提取完成: {audio_path}")
            return str(audio_path)

        video.close()
        logger.info("视频没有音频轨道")
        return None
    except Exception as e:
        logger.error(f"提取音频失败: {e}")
        return None


def call_qwen_image_edit(
    image_path: str,
    prompt: str,
    api_key: str,
    output_path: str,
    size: str | None = None,
) -> str:
    """
    调用Qwen Image Edit API编辑图片

    Args:
        image_path: 输入图片路径
        prompt: 编辑指令
        api_key: API密钥
        output_path: 输出图片路径
        size: 输出尺寸 (如 "1024*1024")

    Returns:
        输出图片路径
    """
    import dashscope
    from dashscope import MultiModalConversation
    import requests

    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    # 编码图片
    image_base64 = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_base64},
                {"text": prompt}
            ]
        }
    ]

    # 构建请求参数
    kwargs = {
        "api_key": api_key,
        "model": "qwen-image-edit-plus",
        "messages": messages,
        "stream": False,
        "n": 1,
        "watermark": False,
        "prompt_extend": True,
    }

    if size:
        kwargs["size"] = size

    logger.debug(f"调用Qwen API处理图片: {image_path}")
    response = MultiModalConversation.call(**kwargs)

    if response.status_code == 200:
        # 获取生成的图片URL
        image_url = response.output.choices[0].message.content[0]['image']
        logger.debug(f"API返回成功，正在下载图片...")

        # 下载图片
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(img_response.content)
            logger.debug(f"图片保存完成: {output_path}")
            return output_path
        else:
            logger.error(f"下载图片失败: HTTP {img_response.status_code}")
            raise Exception(f"下载图片失败: HTTP {img_response.status_code}")
    else:
        logger.error(f"API调用失败: {response.code} - {response.message}")
        raise Exception(f"API调用失败: {response.code} - {response.message}")


def process_frames_parallel(
    frame_paths: list[str],
    prompt: str,
    api_key: str,
    output_dir: Path,
    size: str | None = None,
    max_workers: int = 3,
    progress_callback=None,
) -> list[str]:
    """
    并行处理多个帧

    Args:
        frame_paths: 原始帧路径列表
        prompt: 编辑指令
        api_key: API密钥
        output_dir: 输出目录
        size: 输出尺寸
        max_workers: 最大并行数
        progress_callback: 进度回调函数

    Returns:
        编辑后的帧路径列表
    """
    logger.info(f"开始并行处理帧: 共 {len(frame_paths)} 帧, 并行数: {max_workers}")
    edited_dir = output_dir / "edited_frames"
    edited_dir.mkdir(exist_ok=True)

    edited_paths = [None] * len(frame_paths)
    completed = 0
    total = len(frame_paths)

    def process_single(args):
        idx, frame_path = args
        output_path = str(edited_dir / f"edited_{idx:06d}.png")
        try:
            result = call_qwen_image_edit(frame_path, prompt, api_key, output_path, size)
            return idx, result, None
        except Exception as e:
            # 如果API调用失败，使用原图
            shutil.copy(frame_path, output_path)
            return idx, output_path, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, (i, path)): i
                   for i, path in enumerate(frame_paths)}

        for future in as_completed(futures):
            idx, result_path, error = future.result()
            edited_paths[idx] = result_path
            completed += 1

            if error:
                logger.warning(f"帧 {idx} 处理出错: {error}")
            else:
                logger.info(f"帧 {idx} 处理完成 ({completed}/{total})")

            if progress_callback:
                progress_callback(completed / total, f"已处理 {completed}/{total} 帧")

    logger.info(f"所有帧处理完成: {completed}/{total}")
    return edited_paths


def create_pan_effect_clip(
    image_path: str,
    duration: float,
    target_size: tuple[int, int],
    pan_range: float,
    fps: float,
) -> "ImageClip":
    """
    为单张图片创建平移动效的视频片段

    Args:
        image_path: 图片路径
        duration: 片段时长（秒）
        target_size: 目标尺寸 (宽度, 高度)
        pan_range: 平移范围比例（如 0.05 表示 5%）
        fps: 帧率

    Returns:
        带平移动效的视频片段
    """
    from moviepy import ImageClip

    # 加载原图并放大
    img = Image.open(image_path)
    original_size = img.size

    # 计算放大后的尺寸（放大 pan_range * 2 以确保有足够的移动空间）
    scale_factor = 1 + pan_range * 2
    enlarged_size = (
        int(target_size[0] * scale_factor),
        int(target_size[1] * scale_factor),
    )

    # 放大图片
    img_enlarged = img.resize(enlarged_size, Image.Resampling.LANCZOS)
    img.close()

    # 保存放大后的临时图片
    temp_dir = Path(image_path).parent
    temp_path = temp_dir / f"_temp_enlarged_{Path(image_path).stem}.png"
    img_enlarged.save(temp_path)
    img_enlarged.close()

    # 计算最大偏移量
    max_offset_x = enlarged_size[0] - target_size[0]
    max_offset_y = enlarged_size[1] - target_size[1]

    # 随机选择平移方向：0=左到右, 1=右到左, 2=上到下, 3=下到上
    direction = random.randint(0, 3)

    # 设置起始和结束位置
    if direction == 0:  # 左到右
        start_x, end_x = 0, max_offset_x
        start_y = end_y = max_offset_y // 2
    elif direction == 1:  # 右到左
        start_x, end_x = max_offset_x, 0
        start_y = end_y = max_offset_y // 2
    elif direction == 2:  # 上到下
        start_x = end_x = max_offset_x // 2
        start_y, end_y = 0, max_offset_y
    else:  # 下到上
        start_x = end_x = max_offset_x // 2
        start_y, end_y = max_offset_y, 0

    # 创建图片clip
    clip = ImageClip(str(temp_path)).with_duration(duration)

    # 定义裁剪函数实现平移效果
    def make_frame(get_frame, t):
        # 计算当前时间的进度（0到1）
        progress = t / duration if duration > 0 else 0
        # 线性插值计算当前偏移
        current_x = int(start_x + (end_x - start_x) * progress)
        current_y = int(start_y + (end_y - start_y) * progress)
        # 获取当前帧并裁剪
        frame = get_frame(t)
        cropped = frame[current_y:current_y + target_size[1], current_x:current_x + target_size[0]]
        return cropped

    # 应用平移效果
    clip = clip.transform(make_frame)

    # 删除临时文件
    if temp_path.exists():
        temp_path.unlink()

    return clip


def create_video_from_frames(
    frame_paths: list[str],
    output_path: str,
    fps: float,
    audio_path: str | None = None,
    pan_range: float = 0.0,
) -> str:
    """
    从帧序列创建视频

    Args:
        frame_paths: 帧文件路径列表
        output_path: 输出视频路径
        fps: 帧率
        audio_path: 音频文件路径（可选）
        pan_range: 平移动效范围比例（如 0.05 表示 5%），0 表示无动效

    Returns:
        输出视频路径
    """
    pan_enabled = pan_range > 0
    logger.info(f"开始创建视频: {len(frame_paths)} 帧, {fps}fps, 平移动效: {'启用 ' + str(int(pan_range*100)) + '%' if pan_enabled else '关闭'}")
    from moviepy import ImageSequenceClip, AudioFileClip, concatenate_videoclips

    # 检查并统一所有帧的尺寸
    logger.info("检查帧尺寸...")
    frame_sizes = []
    for path in frame_paths:
        img = Image.open(path)
        frame_sizes.append(img.size)
        img.close()

    # 使用第一帧的尺寸作为目标尺寸
    target_size = frame_sizes[0]
    size_mismatch = any(size != target_size for size in frame_sizes)

    if size_mismatch:
        logger.warning(f"检测到帧尺寸不一致，将统一调整为 {target_size[0]}x{target_size[1]}")
        for i, (path, size) in enumerate(zip(frame_paths, frame_sizes)):
            if size != target_size:
                logger.debug(f"调整帧 {i} 尺寸: {size} -> {target_size}")
                img = Image.open(path)
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(path)
                img.close()
                img_resized.close()
        logger.info("帧尺寸统一完成")

    # 创建视频
    if pan_enabled:
        # 使用平移动效：为每张图片创建带平移效果的clip，然后拼接
        logger.info("正在为每帧创建平移动效...")
        # 计算每帧的持续时间
        frame_duration = 1.0 / fps
        clips = []
        for i, frame_path in enumerate(frame_paths):
            pan_clip = create_pan_effect_clip(
                frame_path,
                duration=frame_duration,
                target_size=target_size,
                pan_range=pan_range,
                fps=fps,
            )
            clips.append(pan_clip)
            if (i + 1) % 10 == 0:
                logger.debug(f"已处理 {i + 1}/{len(frame_paths)} 帧的平移动效")
        # 拼接所有clip
        clip = concatenate_videoclips(clips, method="compose")
        logger.info(f"平移动效视频片段创建完成, 时长: {clip.duration:.2f}秒")
    else:
        # 无平移动效，使用原有方式
        clip = ImageSequenceClip(frame_paths, fps=fps)
        logger.info(f"视频片段创建完成, 时长: {clip.duration:.2f}秒")

    # 添加音频
    audio = None
    if audio_path and os.path.exists(audio_path):
        logger.info(f"正在添加音频: {audio_path}")
        audio = AudioFileClip(audio_path)
        # 确保音频长度与视频匹配
        if audio.duration > clip.duration:
            audio = audio.subclipped(0, clip.duration)
        clip = clip.with_audio(audio)
        logger.info("音频添加完成")

    # 输出视频
    logger.info(f"正在编码输出视频: {output_path}")
    clip.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        logger=None,
    )

    clip.close()
    if audio is not None:
        audio.close()

    logger.info(f"视频创建完成: {output_path}")
    return output_path


def process_video(
    video_path: str,
    interval: float,
    prompt: str,
    api_key: str,
    output_size: str,
    max_workers: int,
    pan_range: float,
    progress=gr.Progress(),
) -> tuple[str, list[tuple[str, str]], str]:
    """
    处理视频的主函数

    Args:
        video_path: 视频文件路径
        interval: 帧提取间隔（秒）
        prompt: 编辑指令
        api_key: API密钥
        output_size: 输出尺寸
        max_workers: 并行处理数
        pan_range: 平移动效范围（0-20%）
        progress: 进度回调

    Returns:
        (输出视频路径, 预览图片列表, 状态消息)
    """
    if not video_path:
        return None, [], "请上传视频文件"

    if not api_key:
        return None, [], "请输入API Key"

    if not prompt:
        return None, [], "请输入编辑指令"

    # 创建工作目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = DEFAULT_OUTPUT_DIR / f"job_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"========== 开始处理视频 ==========")
    logger.info(f"视频路径: {video_path}")
    logger.info(f"工作目录: {work_dir}")
    # pan_range 从 UI 传入的是百分比值（0-20），转换为比例（0-0.20）
    pan_range_ratio = pan_range / 100.0
    pan_range_pct = int(pan_range)
    logger.info(f"参数: 间隔={interval}秒, 并行数={max_workers}, 尺寸={output_size}, 平移动效={pan_range_pct}%")

    try:
        # 步骤1: 提取帧
        logger.info("[步骤1/4] 开始提取视频帧...")
        progress(0.1, desc="正在提取视频帧...")
        frame_paths, fps, (width, height) = extract_frames(video_path, interval, work_dir)

        if not frame_paths:
            logger.error("无法从视频中提取帧")
            return None, [], "无法从视频中提取帧"

        progress(0.15, desc=f"已提取 {len(frame_paths)} 帧")

        # 步骤2: 提取音频
        logger.info("[步骤2/4] 开始提取音频...")
        progress(0.2, desc="正在提取音频...")
        audio_path = extract_audio(video_path, work_dir)

        # 步骤3: 处理帧
        logger.info("[步骤3/4] 开始处理帧...")
        progress(0.25, desc="正在处理帧...")

        # 解析输出尺寸
        size = output_size if output_size and output_size != "原始尺寸" else None

        def update_progress(ratio, msg):
            # 帧处理占25%-85%的进度
            progress(0.25 + ratio * 0.6, desc=msg)

        edited_paths = process_frames_parallel(
            frame_paths,
            prompt,
            api_key,
            work_dir,
            size=size,
            max_workers=max_workers,
            progress_callback=update_progress,
        )

        # 步骤4: 生成视频
        logger.info("[步骤4/4] 开始合成视频...")
        progress(0.9, desc="正在合成视频...")
        output_video_path = str(work_dir / "output.mp4")
        create_video_from_frames(edited_paths, output_video_path, fps, audio_path, pan_range_ratio)

        progress(1.0, desc="处理完成!")

        # 准备预览图片（显示原图和编辑后的对比）
        preview_images = []
        step = max(1, len(frame_paths) // 6)  # 最多显示6组对比
        for i in range(0, len(frame_paths), step):
            if i < len(edited_paths) and edited_paths[i]:
                preview_images.append((frame_paths[i], f"原始帧 {i+1}"))
                preview_images.append((edited_paths[i], f"编辑后 {i+1}"))

        status = f"处理完成! 共处理 {len(frame_paths)} 帧，输出视频: {output_video_path}"
        logger.info(f"========== 视频处理完成 ==========")
        logger.info(f"输出文件: {output_video_path}")
        return output_video_path, preview_images, status

    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        return None, [], f"处理失败: {str(e)}"


def create_ui():
    """创建Gradio界面"""

    with gr.Blocks(title="视频风格编辑器", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎬 视频风格编辑器

        使用通义千问 Qwen Image Edit API 对视频进行风格转换。
        上传视频后，会按指定时间间隔提取帧，使用AI编辑每一帧，然后重新合成视频。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📁 输入设置")

                video_input = gr.Video(
                    label="上传视频",
                    sources=["upload"],
                )

                api_key_input = gr.Textbox(
                    label="DashScope API Key",
                    placeholder="请输入您的API Key",
                    type="password",
                    value=os.getenv("DASHSCOPE_API_KEY", ""),
                )

                prompt_input = gr.Textbox(
                    label="编辑指令",
                    placeholder="描述您想要的视角/景别变化，例如：将画面转换为俯视角度，增加景深效果",
                    lines=3,
                    value="将画面转换为电影级别的广角镜头效果，增强景深和空间感",
                )

                with gr.Row():
                    interval_input = gr.Slider(
                        label="帧提取间隔（秒）",
                        minimum=0.1,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        info="间隔越小，帧数越多，处理时间越长",
                    )

                    workers_input = gr.Slider(
                        label="并行处理数",
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        info="同时处理的帧数，建议2-3",
                    )

                size_input = gr.Dropdown(
                    label="输出尺寸",
                    choices=["原始尺寸", "512*512", "768*768", "1024*1024", "1024*768", "768*1024"],
                    value="原始尺寸",
                    info="设置编辑后图片的分辨率",
                )

                pan_range_input = gr.Slider(
                    label="平移动效范围（%）",
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    info="图片平移范围，0表示关闭，5-10%效果较自然，方向随机",
                )

                process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 📺 输出结果")

                status_output = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                )

                video_output = gr.Video(
                    label="输出视频",
                )

        # 预览区域
        gr.Markdown("### 🖼️ 帧预览（原图 vs 编辑后）")
        preview_gallery = gr.Gallery(
            label="帧对比预览",
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
        )

        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ## 使用步骤

            1. **获取API Key**: 前往 [阿里云百炼平台](https://bailian.console.alibabacloud.com/) 注册并获取 DashScope API Key
            2. **上传视频**: 支持常见视频格式（MP4, AVI, MOV等）
            3. **设置参数**:
               - **帧提取间隔**: 建议0.5-2秒，间隔越小效果越流畅但处理时间越长
               - **编辑指令**: 描述您想要的视觉效果变化
               - **并行处理数**: 建议2-3，过高可能触发API限流
            4. **开始处理**: 点击按钮后等待处理完成
            5. **查看结果**: 预览编辑后的帧并下载输出视频

            ## 编辑指令示例

            - 将画面转换为俯视角度
            - 增加电影级别的景深效果
            - 转换为广角镜头视角
            - 增强画面的空间层次感
            - 将近景转换为中景构图
            - 添加柔和的背景虚化效果

            ## 注意事项

            - 视频较长时处理时间可能较长，请耐心等待
            - API调用会产生费用，请注意控制帧数
            - 建议先用短视频测试效果
            """)

        # 绑定事件
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                interval_input,
                prompt_input,
                api_key_input,
                size_input,
                workers_input,
                pan_range_input,
            ],
            outputs=[video_output, preview_gallery, status_output],
            show_progress=True,
        )

    return app


def main():
    """主函数"""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
