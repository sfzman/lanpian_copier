# Video Style Editor

使用通义千问 Qwen Image Edit API 对视频进行风格转换的 Gradio 应用。

## 功能特性

- 上传视频并按指定时间间隔提取帧
- 使用 Qwen Image Edit API 对每帧进行 AI 编辑（改变视角、景别等）
- 自动提取原视频音频并合并到新视频
- 并行处理加速帧编辑
- 实时预览编辑前后的帧对比
- 支持自定义输出分辨率

## 环境要求

- Python 3.10+
- FFmpeg（用于音视频处理）

## 安装

### 1. 安装 FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
下载 [FFmpeg](https://ffmpeg.org/download.html) 并添加到系统 PATH。

### 2. 安装 uv（如果尚未安装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 克隆项目并安装依赖

```bash
git clone <repository-url>
cd lanpian_copier

# 使用 uv 创建虚拟环境并安装依赖
uv sync
```

## 配置

### 获取 API Key

1. 访问 [阿里云百炼平台](https://bailian.console.alibabacloud.com/)
2. 注册/登录账号
3. 在控制台获取 DashScope API Key

### 设置 API Key（可选）

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```
DASHSCOPE_API_KEY=your_api_key_here
```

或者直接在应用界面中输入 API Key。

## 运行

```bash
# 使用 uv 运行
uv run python app.py

# 或者激活虚拟环境后运行
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
python app.py
```

应用将在 `http://localhost:7860` 启动。

## 使用说明

1. **上传视频**: 支持 MP4, AVI, MOV 等常见格式
2. **输入 API Key**: 如果未在环境变量中配置
3. **设置参数**:
   - **帧提取间隔**: 0.1-5秒，建议 0.5-2 秒
   - **编辑指令**: 描述想要的视觉效果
   - **并行处理数**: 1-5，建议 2-3
   - **输出尺寸**: 可选择固定尺寸或保持原始尺寸
4. **点击处理**: 等待处理完成
5. **下载结果**: 预览并下载输出视频

## 编辑指令示例

```
将画面转换为电影级别的广角镜头效果，增强景深和空间感
将画面转换为俯视角度
增加柔和的背景虚化效果
转换为中景构图，突出主体
添加赛博朋克风格的视觉效果
```

## 项目结构

```
lanpian_copier/
├── app.py              # 主应用文件
├── pyproject.toml      # 项目配置和依赖
├── README.md           # 说明文档
├── .env.example        # 环境变量示例
├── .gitignore          # Git 忽略文件
└── output/             # 输出目录（自动创建）
    └── job_YYYYMMDD_HHMMSS/
        ├── original_frames/   # 原始提取的帧
        ├── edited_frames/     # 编辑后的帧
        ├── audio.mp3          # 提取的音频
        └── output.mp4         # 最终输出视频
```

## 注意事项

- API 调用会产生费用，请注意控制帧数
- 视频较长时处理时间可能较长
- 建议先用短视频（5-10秒）测试效果
- 并行数过高可能触发 API 限流

## 费用说明

Qwen Image Edit API 按生成的图片数量计费。具体价格请参考 [阿里云百炼定价](https://help.aliyun.com/zh/model-studio/billing-overview)。

## 许可证

MIT License
