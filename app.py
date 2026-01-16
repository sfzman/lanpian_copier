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

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

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

    return frame_paths, fps, (width, height)


def extract_audio(video_path: str, output_dir: Path) -> str | None:
    """从视频中提取音频"""
    try:
        from moviepy import VideoFileClip

        audio_path = output_dir / "audio.mp3"
        video = VideoFileClip(video_path)

        if video.audio is not None:
            video.audio.write_audiofile(str(audio_path), logger=None)
            video.close()
            return str(audio_path)

        video.close()
        return None
    except Exception as e:
        print(f"提取音频失败: {e}")
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

    response = MultiModalConversation.call(**kwargs)

    if response.status_code == 200:
        # 获取生成的图片URL
        image_url = response.output.choices[0].message.content[0]['image']

        # 下载图片
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(img_response.content)
            return output_path
        else:
            raise Exception(f"下载图片失败: HTTP {img_response.status_code}")
    else:
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
                print(f"帧 {idx} 处理出错: {error}")

            if progress_callback:
                progress_callback(completed / total, f"已处理 {completed}/{total} 帧")

    return edited_paths


def create_video_from_frames(
    frame_paths: list[str],
    output_path: str,
    fps: float,
    audio_path: str | None = None,
) -> str:
    """
    从帧序列创建视频

    Args:
        frame_paths: 帧文件路径列表
        output_path: 输出视频路径
        fps: 帧率
        audio_path: 音频文件路径（可选）

    Returns:
        输出视频路径
    """
    from moviepy import ImageSequenceClip, AudioFileClip

    # 创建视频
    clip = ImageSequenceClip(frame_paths, fps=fps)

    # 添加音频
    if audio_path and os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        # 确保音频长度与视频匹配
        if audio.duration > clip.duration:
            audio = audio.subclipped(0, clip.duration)
        clip = clip.with_audio(audio)

    # 输出视频
    clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        logger=None,
    )

    clip.close()
    if audio_path and os.path.exists(audio_path):
        audio.close()

    return output_path


def process_video(
    video_path: str,
    interval: float,
    prompt: str,
    api_key: str,
    output_size: str,
    max_workers: int,
    progress=gr.Progress(),
) -> tuple[str, list[tuple[str, str]], str]:
    """
    处理视频的主函数

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

    try:
        # 步骤1: 提取帧
        progress(0.1, desc="正在提取视频帧...")
        frame_paths, fps, (width, height) = extract_frames(video_path, interval, work_dir)

        if not frame_paths:
            return None, [], "无法从视频中提取帧"

        progress(0.15, desc=f"已提取 {len(frame_paths)} 帧")

        # 步骤2: 提取音频
        progress(0.2, desc="正在提取音频...")
        audio_path = extract_audio(video_path, work_dir)

        # 步骤3: 处理帧
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
        progress(0.9, desc="正在合成视频...")
        output_video_path = str(work_dir / "output.mp4")
        create_video_from_frames(edited_paths, output_video_path, fps, audio_path)

        progress(1.0, desc="处理完成!")

        # 准备预览图片（显示原图和编辑后的对比）
        preview_images = []
        step = max(1, len(frame_paths) // 6)  # 最多显示6组对比
        for i in range(0, len(frame_paths), step):
            if i < len(edited_paths) and edited_paths[i]:
                preview_images.append((frame_paths[i], f"原始帧 {i+1}"))
                preview_images.append((edited_paths[i], f"编辑后 {i+1}"))

        status = f"处理完成! 共处理 {len(frame_paths)} 帧，输出视频: {output_video_path}"
        return output_video_path, preview_images, status

    except Exception as e:
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
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        info="间隔越小，帧数越多，处理时间越长",
                    )

                    workers_input = gr.Slider(
                        label="并行处理数",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        info="同时处理的帧数，建议2-3",
                    )

                size_input = gr.Dropdown(
                    label="输出尺寸",
                    choices=["原始尺寸", "512*512", "768*768", "1024*1024", "1024*768", "768*1024"],
                    value="原始尺寸",
                    info="设置编辑后图片的分辨率",
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
