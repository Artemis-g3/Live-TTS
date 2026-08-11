# Live-TTS (配音软件)

> [简体中文](README.md) | **English**

An AI voice dubbing workbench for character dubbing: starting from raw audio, it runs through speaker filtering, speech recognition, emotion labeling, and reference-audio retrieval, then synthesizes dubbing matched to the target lines and emotions using cloud TTS or local VoxCPM2.

The project uses a **PyQt6 desktop GUI + Python backend CLI** architecture. All backend capabilities are invoked by the GUI as subprocesses through `GUI/backend/workflow_cli.py`, so most features can also be used from the command line directly.

---

## Highlights

- **Speaker filtering**: Uses the CAM++ speaker-verification model to score raw clips against the role's reference audio centroid, automatically classifying them into `confirmed / review / reject`.
- **Speech recognition (ASR)**: Calls Qwen3-ASR-Flash; supports Chinese, English, and Japanese, with automatic translation of non-Chinese lines into Simplified Chinese.
- **Emotion labeling**: The Qwen3.5-Omni multimodal model listens to each clip and produces a director-style natural-language voice description; a text model then extracts structured keywords (tone/emotion + prosody/delivery) to build a role voice library.
- **Two-stage retrieval**: Stage one uses `qwen3-rerank` for keyword pre-ranking; stage two uses a text model (Qwen3.6-Flash / DeepSeek) for final ranking based on descriptions and keywords, selecting the reference clips that best match the target dubbing description.
- **Dual-backend TTS**:
  - Cloud: Qwen voice enrollment (`qwen-voice-enrollment`) + `qwen3-tts-vc` synthesis;
  - Local: VoxCPM2 (basic clone / hifi clone), fully offline, with optional style guidance.
- **Long-text dubbing**: An LLM splits long text into emotion-coherent segments and generates voice-guidance keywords for each segment; double-click a segment to jump to the dubbing workbench.
- **Auto guidance**: One click generates retrieval voice/emotion keywords from the target lines.
- **Output management**: Every synthesis run produces a normalized `tts_runs` directory (with `run_summary.json`, reference audio/text, and synthesized audio), supporting listing, playback, deletion, cleanup, and filename normalization.
- **Legacy migration**: One-click migration of the old folder layout (raw/reference/filtered audio, ASR, emotion labeling, etc.) and Excel role libraries into the new `workspace` structure.

---

## UI Preview

### Input Audio Overview

![Input Audio Overview](docs/screenshots/01-input-overview.png)

Scans audio from `input_audio/<role>/` into the role workspace — the starting point of the whole pipeline. The toolbar provides:

- Role selector + "Sync Input Folder": incrementally scans audio per role (wav / mp3 / m4a / flac / ogg, subdirectories supported);
- "Save Reference": marks the selected clips as reference audio for the target role (required by filtering and retrieval; at least 2 clips);
- "Move to Trash": moves unwanted clips into `workspace/trash/`;
- "Duration Stats": bucket statistics across 0-2s / 2-5s / 5-10s / 10-15s / 15-30s / 30s+.

The table shows each clip's file name, duration, sample rate, filter status, and reference marker; the duration distribution is shown below; the built-in player can preview any selected clip.

### Audio Filter

![Audio Filter](docs/screenshots/02-audio-filter.png)

Uses the CAM++ speaker-verification model to compare every candidate against the reference centroid and automatically pick out the target speaker. Layout:

- Top: role selector and "Load Filter";
- Parameters: confirm threshold (default 0.72) and review threshold (default 0.60); review must be lower than confirm;
- Buttons: "Run Filter", "Stop", "Save Results", "Clear Log";
- Results: side-by-side Confirm / Review tables with manual adjustments ("Move to Review", "Add to Confirm", "Exclude");
- Bottom: run log with progress and per-category counts.

Decision logic: `strong = min(centroid similarity, max reference similarity)` reaching the confirm threshold → confirmed; `soft = min(max reference, max(centroid, median reference))` reaching the review threshold → review; otherwise → reject. Click "Save Results" to write `audio_manifest.csv`; only `confirmed` clips are processed by ASR afterwards.

### Speech Recognition

![Speech Recognition](docs/screenshots/03-speech-recognition.png)

Calls Qwen3-ASR-Flash to transcribe the `confirmed` clips and translates non-Chinese lines. Layout:

- Top: role selector and source language (Chinese / English / Japanese);
- Buttons: "Full ASR", "Single ASR", "Full Translation", "Single Translation", "Stop", "Load ASR", "Save All", "Clear Log";
- Middle table: audio file, source language, original text, Chinese text, ASR error, translation error — directly editable for proofreading;
- Bottom: run log with progress and API token usage.

Chinese clips use the recognition result directly as Chinese text; English/Japanese clips are first transcribed into the original text, then translated into Simplified Chinese. After review, "Save All" writes to `workflow_db.csv` (results stay in the table preview until saved).

### Emotion Labeling

![Emotion Labeling](docs/screenshots/04-emotion-labeling.png)

Produces a director-style natural-language voice description plus structured keywords for every clip, building the role voice library. Layout:

- Buttons: "Full/Single Emotion Description", "Full/Single Extract Keywords", "Stop", "Load Emotion Labeling", "Save Emotion Labeling", "Clear Log";
- Middle table: index, audio file, transcript, audio path, natural-language description, tone/emotion, prosody/delivery, keywords — manually editable;
- Bottom: run log with progress and token usage for both steps.

Two-step flow: Step 1 uses the Qwen3.5-Omni multimodal model to listen to the clip and generate a 50-150 character natural-language description; Step 2 uses a text model to extract keywords (tone/emotion/attitude + prosody/delivery) from the description. Every run leaves a JSONL record and summary under `metadata/emotion_runs/`; saving writes to `workflow_db.csv`.

### Dubbing Workbench

![Dubbing Workbench](docs/screenshots/05-dubbing-workbench.png)

The core synthesis page: enter lines → retrieve reference clips → select references → synthesize. The form includes:

- Role and synthesis backend (`api` cloud / `voxcpm2_local_basic` / `voxcpm2_local_hifi`) plus synthesis language (Chinese / English / Japanese);
- "Target Lines": the text to synthesize;
- "Retrieval Voice Guidance": describes the desired voice/emotion (e.g., "gentle, slightly resigned"); the "Auto Guidance" button generates keywords from the lines in one click;
- "Synthesis Voice Guidance": only affects VoxCPM2 Basic, injected into the text in parentheses;
- Parameters: "Second-Round Clip Count" (retrieval top-k, default 10), "Diffusion Steps" and "Imitation Similarity" (VoxCPM2 inference_timesteps / cfg_value).

Below are the retrieval result table and the run log. Workflow: "Start Retrieval" runs the two-stage search (rerank pre-ranking → text-model final ranking) → check the desired reference clips → "Start Synthesis". Outputs are written to `output/tts_runs/<timestamp>/`, and the log prints every artifact path and token usage.

### Long-Text Dubbing

![Long-Text Dubbing](docs/screenshots/06-long-text-dubbing.png)

For dubbing long text (hundreds of characters or more), splitting it into short segments suitable for line-by-line performance. Layout:

- Input area: paste or type long text (poetry, prose, scripts, etc.);
- Controls: "Auto Split & Describe" button plus a "Thinking Mode" checkbox (enables DeepSeek thinking / Qwen enable_thinking);
- Results: a segment table with "Segment Text" and "Voice Guidance" columns;
- Bottom: run log.

An LLM splits the text into 15-60 character segments according to emotion/tone changes, keeping the original wording intact, and generates 2-5 voice-guidance keywords per segment. **Double-click any segment** to jump to the dubbing workbench with the segment and its guidance prefilled for synthesis.

### Output Overview

![Output Overview](docs/screenshots/07-output-overview.png)

Central management of all synthesis outputs for the role. Table columns:

- Audio name: the synthesized file (`synthesized_<timestamp>.wav/mp3`);
- Dubbed lines: the target text of the run;
- Retrieval voice guidance and synthesis voice guidance (shown only for VoxCPM2 Basic).

Supported actions: double-click or select a row to preview, "Open Output Folder", "Delete Selected" removes the corresponding `tts_runs` directory, "Clean Non-conforming" removes legacy directories that violate the naming rules, and "Normalize Filenames" aligns older outputs with the new convention. Each run directory also contains `run_summary.json`, `retrieval_result.json`, reference audio/text, and other artifacts.

### Settings

![Settings](docs/screenshots/08-settings.png)

Central configuration for the runtime environment, API keys, and models — complete this page before first use. It includes:

- **VoxCPM2 model directory**: where the local TTS weights live (default `VoxCPM2`);
- **Runtime environment**: the Python environment used by backend subprocesses, auto-detected from conda/venv directories or chosen manually;
- **Qwen API Key** and **DeepSeek API Key**: required by ASR, emotion labeling, retrieval, and cloud TTS;
- **Model dropdowns**: ASR model, audio description model, retrieval ranking model, text model, voice enrollment model, and cloud synthesis model — all switchable;
- "Save Settings": writes to `voice_gui_settings.json` (excluded by `.gitignore`, never uploaded).

---

## Tech Stack & Models

### Tech Stack

- Python 3.12 (used for the packaged environment)
- PyQt6 desktop UI
- Local inference: PyTorch / torchaudio / modelscope / voxcpm
- Cloud services: DashScope (Alibaba Cloud Bailian, Qwen family) and DeepSeek API
- Audio processing: soundfile / librosa / scipy / numpy
- Persistence: CSV (`audio_manifest.csv`, `workflow_db.csv`) + JSON (run summaries, caches, run records)

### Models

| Purpose | Model | Runtime |
| --- | --- | --- |
| Speaker verification | CAM++ `speech_campplus_sv_zh-cn_16k-common` | Local (shipped in the repo) |
| Speech recognition | `qwen3-asr-flash` | DashScope API |
| Audio emotion description | `qwen3.5-omni-plus` / `qwen3.5-omni-flash` | DashScope API |
| Retrieval re-ranking | `qwen3-rerank` | DashScope API |
| Final ranking / splitting / auto guidance / translation | `qwen3.6-flash` or `deepseek-v4-flash` / `deepseek-v4-pro` | DashScope / DeepSeek API |
| Cloud voice enrollment | `qwen-voice-enrollment` | DashScope API |
| Cloud speech synthesis | `qwen3-tts-vc-2026-01-22` | DashScope API |
| Local speech synthesis | VoxCPM2 (2B, 30 languages, 48 kHz) | Local (weights downloaded separately) |

> All model names can be changed in the GUI Settings page; the values above are the defaults.

---

## Directory Structure

```text
Live-TTS/
├── GUI/                        # PyQt6 desktop frontend
│   ├── main.py                 # Entry point (supports --smoke-test)
│   ├── config.py               # Config load/save, environment variables
│   ├── ui/main_window.py       # Main window: 8 tabs + audio player
│   └── backend/workflow_cli.py # Backend CLI (invoked by the GUI as a subprocess)
├── code/voice_modules/         # Backend feature modules (usable standalone)
│   ├── audio_filter/           # Speaker filtering (CAM++)
│   ├── input_audio/            # Input audio scanning / sync / duration stats
│   ├── common/                 # Audio manifest, workflow DB, migration, state
│   ├── speech_recognition/     # ASR
│   ├── text_translation/       # Line translation
│   ├── emotion_labeling/       # Emotion description + keyword extraction
│   ├── dubbing/                # Retrieval, TTS synthesis, auto guidance, output overview, legacy scripts
│   ├── long_text/              # Long-text splitting + voice guidance
│   └── role_library/           # Role voice library
├── VoxCPM2/                    # Local TTS model directory (weights not committed)
├── build/VoiceDubbingGUI.spec  # PyInstaller spec
├── requirements.txt            # Python dependencies
├── voice_gui_settings.example.json  # Config template
├── input_audio/                # Input audio (local data, not committed)
├── workspace/                  # Role workspace (local data, not committed)
└── voice_gui_runtime/          # GUI runtime state (local data, not committed)
```

---

## Quick Start

### 1. Requirements

- Windows 10/11
- Python 3.12 (recommended)
- NVIDIA GPU + CUDA 12.8 driver when using local VoxCPM2

### 2. Create an Environment and Install Dependencies

```powershell
cd Live-TTS
python -m venv .venv
.\.venv\Scripts\activate
```

Install PyTorch matching your CUDA version first (the current environment uses CUDA 12.8):

```powershell
pip install torch==2.10.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

> For CPU-only or other CUDA versions, choose the matching wheel from the official PyTorch index.

### 3. Configure API Keys and Runtime Environment

```powershell
$env:DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
$env:DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"   # only needed for DeepSeek ranking models
```

You can also fill them in on the GUI Settings page (saved to `voice_gui_settings.json`, which is excluded by `.gitignore` — never upload it).

### 4. Launch

```powershell
python GUI\main.py
```

Smoke test (prints the window title and role list, then exits):

```powershell
python GUI\main.py --smoke-test
```

### 5. Prepare the Local VoxCPM2 Model (Optional)

The public repo only contains VoxCPM2 configuration and tokenizer files, **not the model weights**. To use local dubbing, place the following files in `VoxCPM2/`:

```text
VoxCPM2/model.safetensors
VoxCPM2/audiovae.pth
```

They are not needed if you only use the cloud TTS backend.

---

## Usage Workflow

A typical character dubbing workflow:

1. Launch the GUI and configure the runtime environment, API keys, and models on the Settings page.
2. Put the role's raw audio into `input_audio/<role>/` (subdirectories are supported).
3. On the "Input Audio Overview" tab, select the role and click "Sync Input Folder".
4. Select 2+ reference clips of the target role and click "Save Reference".
5. On the "Audio Filter" tab, set the thresholds and run the filter; review and save the results.
6. On the "Speech Recognition" tab, run full ASR (and translation for non-Chinese audio), then save.
7. On the "Emotion Labeling" tab, run full emotion description and keyword extraction, then save.
8. On the "Dubbing Workbench" tab, enter the target lines and retrieval voice guidance, run retrieval, select reference clips and a synthesis backend, then synthesize.
9. Review, listen to, and manage each run on the "Output Overview" tab.

For long text, use the "Long-Text Dubbing" tab: paste the text, split it with guidance, then double-click a segment to jump to the dubbing workbench.

---

## Command Line Usage

The backend is independent of the GUI and exposed through `workflow_cli.py`:

```powershell
python -m GUI.backend.workflow_cli --help
python -m GUI.backend.workflow_cli tts --help
```

Every command prints a structured JSON result prefixed with `VOICE_GUI_RESULT_JSON=`, which makes scripting easy. See [docs/命令行工具.md](docs/命令行工具.md) (Chinese) for the full command reference.

---

## Packaging as exe

```powershell
pip install pyinstaller
python -m PyInstaller --noconfirm --clean --distpath . --workpath build\pyinstaller build\VoiceDubbingGUI.spec
```

This produces `VoiceDubbingGUI.exe` in the project root (the GUI invokes the backend as a subprocess, so the `code/` directory and model directories must still be present at runtime).

---

## Data & Privacy

The software generates the following local data at runtime. All of it is excluded by `.gitignore` — **do not commit it to GitHub**:

```text
voice_gui_settings.json   # Local config (contains API keys)
voice_gui_runtime/        # GUI runtime state
workspace/                # Role audio, metadata, and outputs
input_audio/              # Input audio assets
```

See [docs/数据格式说明.md](docs/数据格式说明.md) (Chinese) for details.

---

## FAQ

**Q: The filter reports "at least 2 reference clips are required"?**

Select at least 2 clips of the target role on the Input Audio Overview tab and click "Save Reference" first.

**Q: ASR / emotion labeling / retrieval reports a missing API key?**

Fill in the Qwen API key on the Settings page (a DeepSeek key is also required for DeepSeek models), or set the corresponding environment variables.

**Q: Local VoxCPM2 synthesis reports that the model directory does not exist?**

Make sure `VoxCPM2/model.safetensors` and `VoxCPM2/audiovae.pth` are in place and confirm the model directory path on the Settings page.

**Q: How do I use only the cloud API?**

Select the `api` backend on the dubbing workbench; no VoxCPM2 weights are needed.

---

## Disclaimer (AI Voice Cloning)

This project includes AI voice-cloning and speech-synthesis capabilities. Please comply with the following before using it:

1. **Lawful use only**: Use it only for cloning your own voice, voices of people who have explicitly authorized you, authorized dubbing production (film / game / audiobook), or research and learning.
2. **No abuse**: Strictly prohibited for impersonation, fraud, defamation, spreading disinformation, deepfakes, or unauthorized imitation of any real person's voice, or any other unlawful or unethical purpose.
3. **The user is responsible for authorization and compliance**: Before using this project, make sure you have lawful sources and authorization for the audio material, and comply with the laws and regulations of your jurisdiction as well as the terms of service of the models and APIs used (DashScope, DeepSeek, VoxCPM2, etc.).
4. **Label AI-generated content**: When publicly distributing AI-generated speech, label it as AI-generated in accordance with local law and platform rules.
5. The project authors are not responsible for any misuse by users or third parties.

---

## Third-Party Licenses & Acknowledgments

This project builds upon the following open-source projects and cloud services:

| Project | Purpose | License / Terms |
| --- | --- | --- |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (OpenBMB) | Local speech synthesis (weights downloaded separately from `openbmb/VoxCPM2`) | Apache-2.0 |
| [CAM++ speaker-verification model](https://github.com/alibaba-damo-academy/3D-Speaker) | Speaker filtering (shipped in the repo) | Apache-2.0 |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | Desktop GUI | GPL-3.0 or commercial (Riverbank) |
| PyInstaller | exe packaging | GPL-2.0 / GPL-3.0 (with bootloader exception) |
| PyTorch / torchaudio | Deep learning runtime | BSD-3-Clause |
| ModelScope | Model loading | Apache-2.0 |
| numpy / pandas / soundfile / librosa / scipy / openpyxl / requests / openai / dashscope, etc. | General dependencies | Each under its own license |

Cloud services:

- **Qwen models** (`qwen3-asr-flash`, `qwen3.5-omni`, `qwen3-rerank`, `qwen3.6-flash`, `qwen-voice-enrollment`, `qwen3-tts-vc`) are accessed via the Alibaba Cloud Bailian DashScope API; usage is subject to Alibaba Cloud's terms of service.
- **DeepSeek API** (`deepseek-v4-*`) usage is subject to DeepSeek's terms of service.

Notes:

- The code in this project itself is released under the MIT license; third-party components remain subject to their own licenses.
- When packaging and distributing binaries with PyQt6, please observe PyQt6's GPL-3.0 obligations or purchase a commercial license from Riverbank.

---

## Documentation

The detailed docs are currently in Chinese:

- [使用手册](docs/使用手册.md) (User manual): step-by-step GUI instructions
- [架构说明](docs/架构说明.md) (Architecture): modules, data flow, and workspace layout
- [命令行工具](docs/命令行工具.md) (CLI reference): all workflow_cli commands
- [数据格式说明](docs/数据格式说明.md) (Data formats): CSV / JSON fields and cache formats
- [开发指南](docs/开发指南.md) (Development guide): environment setup, code organization, packaging, and contribution conventions

## License

[MIT](LICENSE)
