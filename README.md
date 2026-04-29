## 一、下载项目

从 GitHub 下载或克隆项目后，进入项目根目录：

```powershell
cd 配音软件
```

如果你的文件夹名称不是 `配音软件`，请进入你实际下载后的项目目录。

## 二、创建 Python 环境

建议使用 Python 3.12。

在项目根目录执行：

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\activate
```

安装依赖：

```powershell
pip install -r requirement.txt
```

## 三、启动 GUI

在项目根目录执行：

```powershell
python GUI\main.py
```

启动后进入 `设置` 页，配置运行环境和 API Key。

## 四、配置运行环境

在 GUI 的 `设置` 页中找到 `运行环境`。

如果下拉框没有自动找到当前环境，点击右侧 `选择`，选择你的 Python 环境目录。

常见目录示例：

```text
项目目录\.venv\Scripts
```

或者 Conda 环境目录：

```text
C:\Users\你的用户名\.conda\envs\你的环境名
```

软件会检查所选目录下是否存在 `python.exe`，找到后会弹窗提示。

## 五、配置 API Key

如果使用云端 ASR、情感标定、检索排序或云端 TTS，需要填写 API Key。

可以在 GUI 的 `设置` 页填写：

- `Bai Lian API Key`
- `Deepseek API Key`，仅在使用 Deepseek 排序模型时需要

也可以通过环境变量设置：

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_key"
$env:DEEPSEEK_API_KEY="your_deepseek_key"
```

不要把自己的 API Key 上传到 GitHub。

## 六、本地 VoxCPM2 模型

公开仓库只包含 VoxCPM2 的配置和 tokenizer 文件，不包含大模型权重。

如果要使用本地 VoxCPM2 配音，需要自行准备以下文件，并放入 `VoxCPM2/` 目录：

```text
VoxCPM2/model.safetensors
VoxCPM2/audiovae.pth
```

如果只使用云端 API 配音，可以不下载这两个文件。

## 七、导入音频并运行流程

常见使用流程：

1. 启动 GUI。
2. 在设置页配置运行环境和 API Key。
3. 将角色音频放入输入目录，或通过 GUI 导入。
4. 扫描输入音频。
5. 进行说话人筛选。
6. 运行 ASR。
7. 运行情感标定。
8. 在配音工作台填写目标台词和声音检索描述。
9. 点击开始检索。
10. 选择参考音频后合成配音。

软件运行过程中会自动生成：

```text
voice_gui_settings.json
voice_gui_runtime/
workspace/
input_audio/
```

这些都是本地数据，不应提交到 GitHub。

## 八、打包 exe

如果需要打包 Windows exe，先确保当前 Python 环境已安装 PyInstaller。

```powershell
pip install pyinstaller
```

然后在项目根目录执行：

```powershell
python -m PyInstaller --noconfirm --clean --distpath . --workpath build\pyinstaller build\VoiceDubbingGUI.spec
```

打包完成后，根目录会生成：

```text
VoiceDubbingGUI.exe
```

