from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from voice_modules.common.project_state import get_project_state
from GUI.config import (
    ASR_MODEL_OPTIONS,
    CLOUD_TTS_MODEL_OPTIONS,
    EMOTION_MODEL_OPTIONS,
    FINAL_RANKING_TEXT_MODEL_OPTIONS,
    RETRIEVAL_RERANK_MODEL_OPTIONS,
    VOICE_ENROLLMENT_MODEL_OPTIONS,
    AppConfig,
    PROJECT_ROOT,
)
from voice_modules.role_library.role_library import resolve_role_paths


RESULT_PREFIX = "VOICE_GUI_RESULT_JSON="
EMOTION_TONE_KEYWORDS = "情绪关键词"
EMOTION_DELIVERY_KEYWORDS = "技巧关键词"
BACKEND_LABELS = {
    "voxcpm2_local_hifi": "VoxCPM2 Hi-Fi",
    "voxcpm2_local_basic": "VoxCPM2 Basic",
    "api": "百炼 API",
}
SYNTHESIS_LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
}
ASR_SOURCE_LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.config = AppConfig.load()
        self.current_process: QProcess | None = None
        self.current_process_kind = ""
        self.stop_requested = False
        self.process_output = ""
        self.active_log_view: QTextEdit | None = None
        self.last_result: dict[str, Any] | None = None
        self.role_names: list[str] = []
        self.manifest_rows: list[dict[str, Any]] = []
        self.filter_results_dirty = False
        self.asr_rows: list[dict[str, Any]] = []
        self.asr_display_headers: list[str] = []
        self.asr_hidden_headers = {"sha256", "role", "file_name", "file_path", "model", "translation_model"}
        self.emotion_rows: list[dict[str, Any]] = []
        self.emotion_display_headers: list[str] = []
        self.emotion_hidden_headers = {"sha256", "model", "error", "索引", "音频路径", "情绪语气", "音频表达技巧", "关键词"}
        self.retrieval_rows: list[dict[str, Any]] = []
        self.output_overview_rows: list[dict[str, Any]] = []
        self.output_overview_display_headers = ["语音名称", "配音台词", "检索声音指导文本", "合成声音指导文本"]
        self.last_retrieval_result_path = ""
        self.last_session_dir = ""
        self.current_audio_path = ""
        self._player_user_dragging = False
        self.lt_input_edit: QTextEdit = None  # type: ignore[assignment]
        self.lt_split_btn: QPushButton = None  # type: ignore[assignment]
        self.lt_thinking_check: QCheckBox = None  # type: ignore[assignment]
        self.lt_segment_table: QTableWidget = None  # type: ignore[assignment]
        self.lt_log_view: QTextEdit = None  # type: ignore[assignment]

        self.setWindowTitle("配音软件 GUI")
        self.resize(1280, 820)
        self.tabs = QTabWidget()
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.85)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.result_path_edit = QLineEdit()
        self.result_path_edit.setReadOnly(True)

        self._build_central_layout()
        self._build_input_overview_tab()
        self._build_audio_filter_tab()
        self._build_asr_tab()
        self._build_emotion_tab()
        self._build_dubbing_tab()
        self._build_long_text_dubbing_tab()
        self._build_output_overview_tab()
        self._build_settings_tab()
        self._connect_player_signals()
        self.reload_project_state()
        if self.role_names and "--smoke-test" not in sys.argv:
            self.load_output_overview(silent=True)

    def _build_central_layout(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self._build_player_bar())
        self.setCentralWidget(container)

    def _build_player_bar(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(
            """
            QWidget {
                border: 1px solid #d9e1ec;
                border-radius: 8px;
                background: #f7f9fc;
            }
            QSlider::groove:horizontal {
                border: 0;
                height: 6px;
                background: #d8e0ea;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4b8bf4;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 1px solid #4b8bf4;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            """
        )
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.player_title_label = QLabel("未选择音频")
        self.player_title_label.setStyleSheet("font-weight: 600; color: #1f2937;")
        self.player_time_label = QLabel("00:00 / 00:00")
        self.player_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.player_time_label.setStyleSheet("color: #526277;")
        top_row.addWidget(self.player_title_label, 1)
        top_row.addWidget(self.player_time_label)

        self.player_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.player_position_slider.setRange(0, 0)
        self.player_position_slider.setEnabled(False)
        self.player_position_slider.sliderPressed.connect(self._on_player_slider_pressed)
        self.player_position_slider.sliderReleased.connect(self._on_player_slider_released)
        self.player_position_slider.sliderMoved.connect(self._on_player_slider_moved)

        controls = QHBoxLayout()
        self.player_play_pause_btn = QPushButton("播放")
        self.player_play_pause_btn.clicked.connect(self.toggle_player_playback)
        self.player_play_pause_btn.setEnabled(False)
        self.player_play_pause_btn.setMinimumWidth(88)
        self.player_path_label = QLabel("")
        self.player_path_label.setStyleSheet("color: #6b7280;")
        self.player_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.player_play_pause_btn)
        controls.addWidget(self.player_path_label, 1)

        layout.addLayout(top_row)
        layout.addWidget(self.player_position_slider)
        layout.addLayout(controls)
        return widget

    def _connect_player_signals(self) -> None:
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.durationChanged.connect(self._on_player_duration_changed)
        self.player.playbackStateChanged.connect(self._on_player_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)

    def _build_input_overview_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.input_role_combo = QComboBox()
        sync_btn = QPushButton("同步输入文件夹")
        sync_btn.clicked.connect(self.sync_input_audio)
        save_reference_btn = QPushButton("保存参考")
        save_reference_btn.clicked.connect(self.save_reference_selection)
        delete_btn = QPushButton("移入回收区")
        delete_btn.clicked.connect(lambda: self.update_selected_manifest_status("deleted"))
        stats_btn = QPushButton("统计时长分布")
        stats_btn.clicked.connect(self.load_duration_stats)
        for widget in [QLabel("角色"), self.input_role_combo, sync_btn, save_reference_btn, delete_btn, stats_btn]:
            controls.addWidget(widget)
        controls.addStretch(1)

        self.input_table = QTableWidget()
        self.input_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.input_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.input_table.itemSelectionChanged.connect(self.handle_input_table_selection_changed)
        self.duration_table = QTableWidget()
        self.duration_table.setMaximumHeight(190)
        self.configure_table_editing(self.input_table, editable=False)
        self.configure_table_editing(self.duration_table, editable=False)

        root.addLayout(controls)
        root.addWidget(self.input_table, 1)
        root.addWidget(QLabel("时长分布"))
        root.addWidget(self.duration_table)
        self.tabs.addTab(page, "输入音频概览")

    def _build_audio_filter_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        top = QHBoxLayout()
        self.filter_role_combo = QComboBox()
        refresh_btn = QPushButton("加载音频筛选")
        refresh_btn.clicked.connect(self.load_processing_state)
        top.addWidget(QLabel("角色"))
        top.addWidget(self.filter_role_combo)
        top.addWidget(refresh_btn)
        top.addStretch(1)
        root.addLayout(top)

        filter_layout = QVBoxLayout()
        filter_controls = QHBoxLayout()
        self.confirm_threshold_spin = QDoubleSpinBox()
        self.confirm_threshold_spin.setRange(-1.0, 1.0)
        self.confirm_threshold_spin.setDecimals(2)
        self.confirm_threshold_spin.setSingleStep(0.01)
        self.confirm_threshold_spin.setValue(0.72)
        self.review_threshold_spin = QDoubleSpinBox()
        self.review_threshold_spin.setRange(-1.0, 1.0)
        self.review_threshold_spin.setDecimals(2)
        self.review_threshold_spin.setSingleStep(0.01)
        self.review_threshold_spin.setValue(0.60)
        run_filter_btn = QPushButton("运行筛选")
        run_filter_btn.clicked.connect(self.run_filter_audio)
        stop_filter_btn = QPushButton("停止")
        stop_filter_btn.clicked.connect(lambda: self.stop_process("filter", "音频筛选", self.clear_filter_preview))
        save_filter_btn = QPushButton("保存结果")
        save_filter_btn.clicked.connect(self.save_filter_results)
        clear_filter_log_btn = QPushButton("清空日志")
        clear_filter_log_btn.clicked.connect(lambda: self.clear_log_view(self.filter_log_view))
        filter_controls.addWidget(QLabel("confirm"))
        filter_controls.addWidget(self.confirm_threshold_spin)
        filter_controls.addWidget(QLabel("review"))
        filter_controls.addWidget(self.review_threshold_spin)
        filter_controls.addWidget(run_filter_btn)
        filter_controls.addWidget(stop_filter_btn)
        filter_controls.addWidget(save_filter_btn)
        filter_controls.addWidget(clear_filter_log_btn)
        filter_controls.addStretch(1)
        filter_layout.addLayout(filter_controls)

        filter_tables = QHBoxLayout()
        self.confirm_table = QTableWidget()
        self.review_table = QTableWidget()
        for table in [self.confirm_table, self.review_table]:
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.configure_table_editing(table, editable=False)
        self.confirm_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.confirm_table, self.confirm_rows())
        )
        self.review_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.review_table, self.review_rows())
        )
        filter_tables.addWidget(self._wrap_table("Confirm", self.confirm_table, [
            ("移到 Review", lambda: self.move_filter_status(self.confirm_table, self.confirm_rows(), "review")),
            ("移出", lambda: self.move_filter_status(self.confirm_table, self.confirm_rows(), "excluded")),
        ]))
        filter_tables.addWidget(self._wrap_table("Review", self.review_table, [
            ("加入 Confirm", lambda: self.move_filter_status(self.review_table, self.review_rows(), "confirmed")),
            ("移出", lambda: self.move_filter_status(self.review_table, self.review_rows(), "excluded")),
        ]))
        filter_layout.addLayout(filter_tables)
        self.filter_log_view = self.create_task_log_view()
        filter_layout.addWidget(QLabel("运行日志"))
        filter_layout.addWidget(self.filter_log_view)
        root.addLayout(filter_layout)
        self.tabs.addTab(page, "音频筛选")

    def _build_asr_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        top = QHBoxLayout()
        self.asr_role_combo = QComboBox()
        self.asr_language_combo = QComboBox()
        for key, label in ASR_SOURCE_LANGUAGE_LABELS.items():
            self.asr_language_combo.addItem(label, key)
        index = self.asr_language_combo.findData(self.config.asr_source_language)
        self.asr_language_combo.setCurrentIndex(index if index >= 0 else 0)
        top.addWidget(QLabel("角色"))
        top.addWidget(self.asr_role_combo)
        top.addWidget(QLabel("原始语言"))
        top.addWidget(self.asr_language_combo)
        top.addStretch(1)
        root.addLayout(top)

        asr_layout = QVBoxLayout()
        asr_controls = QHBoxLayout()
        run_asr_btn = QPushButton("全量ASR")
        run_asr_btn.clicked.connect(self.run_asr)
        run_single_asr_btn = QPushButton("单条ASR")
        run_single_asr_btn.clicked.connect(self.run_single_asr)
        run_translate_btn = QPushButton("全量翻译")
        run_translate_btn.clicked.connect(self.run_translation)
        run_single_translate_btn = QPushButton("单条翻译")
        run_single_translate_btn.clicked.connect(self.run_single_translation)
        stop_asr_btn = QPushButton("停止")
        stop_asr_btn.clicked.connect(lambda: self.stop_process("asr", "语音识别", self.clear_asr_preview))
        load_asr_btn = QPushButton("加载 ASR")
        load_asr_btn.clicked.connect(self.load_asr)
        save_asr_btn = QPushButton("全部保存")
        save_asr_btn.clicked.connect(self.save_asr)
        clear_asr_log_btn = QPushButton("清空日志")
        clear_asr_log_btn.clicked.connect(lambda: self.clear_log_view(self.asr_log_view))
        for widget in [run_asr_btn, run_single_asr_btn, run_translate_btn, run_single_translate_btn, stop_asr_btn, load_asr_btn, save_asr_btn, clear_asr_log_btn]:
            asr_controls.addWidget(widget)
        asr_controls.addStretch(1)
        self.asr_table = QTableWidget()
        self.asr_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.asr_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.asr_table, self.asr_rows)
        )
        self.configure_table_editing(self.asr_table, editable=True)
        asr_layout.addLayout(asr_controls)
        asr_layout.addWidget(self.asr_table)
        self.asr_log_view = self.create_task_log_view()
        asr_layout.addWidget(QLabel("运行日志"))
        asr_layout.addWidget(self.asr_log_view)
        root.addLayout(asr_layout)
        self.tabs.addTab(page, "语音识别")

    def _build_emotion_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        top = QHBoxLayout()
        self.emotion_role_combo = QComboBox()
        top.addWidget(QLabel("角色"))
        top.addWidget(self.emotion_role_combo)
        top.addStretch(1)
        root.addLayout(top)

        emotion_layout = QVBoxLayout()
        emotion_controls = QHBoxLayout()
        run_desc_btn = QPushButton("全量情感描述")
        run_desc_btn.clicked.connect(self.run_emotion_description)
        run_single_desc_btn = QPushButton("单条情感描述")
        run_single_desc_btn.clicked.connect(self.run_single_emotion_description)
        run_kw_btn = QPushButton("全量提取关键词")
        run_kw_btn.clicked.connect(self.run_emotion_keywords)
        run_single_kw_btn = QPushButton("单条提取关键词")
        run_single_kw_btn.clicked.connect(self.run_single_emotion_keywords)
        stop_emotion_btn = QPushButton("停止")
        stop_emotion_btn.clicked.connect(lambda: self.stop_process("emotion", "情感标定", self.clear_emotion_preview))
        load_emotion_btn = QPushButton("加载情感标定")
        load_emotion_btn.clicked.connect(self.load_emotion)
        save_emotion_btn = QPushButton("保存情感标定")
        save_emotion_btn.clicked.connect(self.save_emotion)
        clear_emotion_log_btn = QPushButton("清空日志")
        clear_emotion_log_btn.clicked.connect(lambda: self.clear_log_view(self.emotion_log_view))
        for widget in [run_desc_btn, run_single_desc_btn, run_kw_btn, run_single_kw_btn, stop_emotion_btn, load_emotion_btn, save_emotion_btn, clear_emotion_log_btn]:
            emotion_controls.addWidget(widget)
        emotion_controls.addStretch(1)
        self.emotion_table = QTableWidget()
        self.emotion_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.emotion_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.emotion_table, self.emotion_rows)
        )
        self.configure_table_editing(self.emotion_table, editable=True)
        emotion_layout.addLayout(emotion_controls)
        emotion_layout.addWidget(self.emotion_table)
        self.emotion_log_view = self.create_task_log_view()
        emotion_layout.addWidget(QLabel("运行日志"))
        emotion_layout.addWidget(self.emotion_log_view)
        root.addLayout(emotion_layout)
        self.tabs.addTab(page, "情感标定")

    def _build_dubbing_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        form_box = QGroupBox("配音工作台")
        form = QFormLayout(form_box)
        self.role_combo = QComboBox()
        self.backend_combo = QComboBox()
        for key, label in BACKEND_LABELS.items():
            self.backend_combo.addItem(label, key)
        index = self.backend_combo.findData(self.config.last_backend)
        self.backend_combo.setCurrentIndex(index if index >= 0 else 0)
        self.synthesis_language_combo = QComboBox()
        for key, label in SYNTHESIS_LANGUAGE_LABELS.items():
            self.synthesis_language_combo.addItem(label, key)
        index = self.synthesis_language_combo.findData(self.config.synthesis_language)
        self.synthesis_language_combo.setCurrentIndex(index if index >= 0 else 0)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setPlaceholderText("请输入要合成的配音台词")
        self.transcript_edit.setFixedHeight(92)
        self.retrieval_guidance_edit = QTextEdit()
        self.retrieval_guidance_edit.setPlaceholderText("请输入用于检索参考音频的声音/情绪指导")
        self.retrieval_guidance_edit.setFixedHeight(68)
        self.synthesis_guidance_edit = QTextEdit()
        self.synthesis_guidance_edit.setPlaceholderText("请输入传递给VoxCPM2 Basic的声音指导，注意指导仅对该模型生效，可留空")
        self.synthesis_guidance_edit.setFixedHeight(68)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        self.top_k_spin.setValue(self.config.reference_window)
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 5000)
        self.gap_spin.setValue(self.config.gap_ms)
        self.voxcpm2_diffusion_steps_spin = QSpinBox()
        self.voxcpm2_diffusion_steps_spin.setRange(1, 200)
        self.voxcpm2_diffusion_steps_spin.setValue(self.config.voxcpm2_diffusion_steps)
        self.voxcpm2_cfg_spin = QDoubleSpinBox()
        self.voxcpm2_cfg_spin.setRange(1.0, 3.0)
        self.voxcpm2_cfg_spin.setDecimals(1)
        self.voxcpm2_cfg_spin.setSingleStep(0.1)
        self.voxcpm2_cfg_spin.setValue(self.config.voxcpm2_cfg_value)
        form.addRow("角色", self.role_combo)
        form.addRow("合成后端", self.backend_combo)
        form.addRow("合成语言", self.synthesis_language_combo)
        form.addRow("配音台词", self.transcript_edit)
        retrieval_guidance_row = QHBoxLayout()
        retrieval_guidance_row.addWidget(self.retrieval_guidance_edit, 1)
        self.auto_guidance_btn = QPushButton("自动指导")
        self.auto_guidance_btn.clicked.connect(self.run_auto_guidance)
        retrieval_guidance_row.addWidget(self.auto_guidance_btn)
        form.addRow("检索声音指导", retrieval_guidance_row)
        form.addRow("合成声音指导", self.synthesis_guidance_edit)
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("第二轮音频数量"))
        params_row.addWidget(self.top_k_spin)
        params_row.addSpacing(30)
        params_row.addWidget(QLabel("扩散轮数"))
        params_row.addWidget(self.voxcpm2_diffusion_steps_spin)
        params_row.addSpacing(30)
        params_row.addWidget(QLabel("模仿相似度"))
        params_row.addWidget(self.voxcpm2_cfg_spin)
        params_row.addStretch(1)
        form.addRow(params_row)
        buttons = QHBoxLayout()
        self.retrieve_btn = QPushButton("开始检索")
        self.retrieve_btn.clicked.connect(self.run_retrieval)
        self.synthesize_btn = QPushButton("开始合成")
        self.synthesize_btn.clicked.connect(self.synthesize_tts)
        self.open_dir_btn = QPushButton("打开输出文件夹")
        self.open_dir_btn.clicked.connect(self.open_role_output_dir)
        clear_main_log_btn = QPushButton("清空日志")
        clear_main_log_btn.clicked.connect(lambda: self.clear_log_view(self.log_view))
        for widget in [self.retrieve_btn, self.synthesize_btn, self.open_dir_btn, clear_main_log_btn]:
            buttons.addWidget(widget)
        buttons.addStretch(1)
        self.retrieval_table = QTableWidget()
        self.retrieval_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.retrieval_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.retrieval_table, self.retrieval_rows)
        )
        self.configure_table_editing(self.retrieval_table, editable=False)
        root.addWidget(form_box)
        root.addLayout(buttons)
        root.addWidget(QLabel("检索到的参考音频"))
        root.addWidget(self.retrieval_table, 1)
        root.addWidget(QLabel("运行日志"))
        root.addWidget(self.log_view, 1)
        self.tabs.addTab(page, "配音工作台")

    def _build_long_text_dubbing_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)

        input_box = QGroupBox("输入文本")
        input_layout = QVBoxLayout(input_box)
        self.lt_input_edit = QTextEdit()
        self.lt_input_edit.setPlaceholderText("请粘贴或输入长文本（诗歌、散文、台词等，支持数百字以上）")
        self.lt_input_edit.setMinimumHeight(150)
        input_layout.addWidget(self.lt_input_edit)

        btn_layout = QHBoxLayout()
        self.lt_split_btn = QPushButton("自动分句描述")
        self.lt_split_btn.clicked.connect(self.run_sentence_split)
        self.lt_thinking_check = QCheckBox("启用思考模式")
        self.lt_thinking_check.setChecked(True)
        btn_layout.addWidget(self.lt_split_btn)
        btn_layout.addWidget(self.lt_thinking_check)
        btn_layout.addStretch(1)
        input_layout.addLayout(btn_layout)
        root.addWidget(input_box)

        results_box = QGroupBox("分句结果")
        results_layout = QVBoxLayout(results_box)
        self.lt_segment_table = QTableWidget()
        self.lt_segment_table.setColumnCount(2)
        self.lt_segment_table.setHorizontalHeaderLabels(["分句文本", "声音指导"])
        self.lt_segment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.lt_segment_table.horizontalHeader().setStretchLastSection(True)
        self.lt_segment_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lt_segment_table.cellDoubleClicked.connect(self._jump_segment_to_dubbing)
        results_layout.addWidget(self.lt_segment_table, 1)
        root.addWidget(results_box, 1)

        root.addWidget(QLabel("运行日志"))
        self.lt_log_view = QTextEdit()
        self.lt_log_view.setReadOnly(True)
        self.lt_log_view.setFixedHeight(120)
        root.addWidget(self.lt_log_view)

        self.tabs.addTab(page, "长文本配音")

    def _build_settings_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        form_box = QGroupBox("设置")
        grid = QGridLayout(form_box)
        self.voice_python_edit = QLineEdit(self.config.voice_python)
        self.voice_python_edit.setReadOnly(True)
        self.runtime_env_combo = QComboBox()
        self.populate_runtime_environment_combo()
        self.runtime_env_combo.currentIndexChanged.connect(self.handle_runtime_environment_changed)
        self.voxcpm_path_edit = QLineEdit(self.config.voxcpm2_model_path)
        self.api_key_edit = QLineEdit(self.config.dashscope_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepseek_api_key_edit = QLineEdit(self.config.deepseek_api_key)
        self.deepseek_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.asr_model_edit = self.create_model_combo(ASR_MODEL_OPTIONS, self.config.asr_model)
        self.emotion_model_edit = self.create_model_combo(EMOTION_MODEL_OPTIONS, self.config.emotion_model)
        self.retrieval_rerank_model_edit = self.create_model_combo(RETRIEVAL_RERANK_MODEL_OPTIONS, self.config.retrieval_rerank_model)
        self.retrieval_text_model_edit = self.create_model_combo(FINAL_RANKING_TEXT_MODEL_OPTIONS, self.config.retrieval_text_model)
        self.voice_enrollment_model_edit = self.create_model_combo(VOICE_ENROLLMENT_MODEL_OPTIONS, self.config.voice_enrollment_model)
        self.tts_target_model_edit = self.create_model_combo(CLOUD_TTS_MODEL_OPTIONS, self.config.tts_target_model)
        rows = [
            ("VoxCPM2 模型目录", self.voxcpm_path_edit, lambda: self.select_dir(self.voxcpm_path_edit)),
        ]
        for row, (label, edit, handler) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            button = QPushButton("选择")
            button.clicked.connect(handler)
            grid.addWidget(button, row, 2)
        grid.addWidget(QLabel("运行环境"), 1, 0)
        grid.addWidget(self.runtime_env_combo, 1, 1)
        runtime_env_button = QPushButton("选择")
        runtime_env_button.clicked.connect(self.select_runtime_environment_dir)
        grid.addWidget(runtime_env_button, 1, 2)
        grid.addWidget(QLabel("Qwen API Key"), 2, 0)
        grid.addWidget(self.api_key_edit, 2, 1, 1, 2)
        grid.addWidget(QLabel("Deepseek API Key"), 3, 0)
        grid.addWidget(self.deepseek_api_key_edit, 3, 1, 1, 2)
        model_rows = [
            ("ASR 模型", self.asr_model_edit),
            ("音频描述模型", self.emotion_model_edit),
            ("检索排序模型", self.retrieval_rerank_model_edit),
            ("文本模型", self.retrieval_text_model_edit),
            ("声音复刻模型", self.voice_enrollment_model_edit),
            ("云端语音合成模型", self.tts_target_model_edit),
        ]
        for offset, (label, edit) in enumerate(model_rows, start=4):
            grid.addWidget(QLabel(label), offset, 0)
            grid.addWidget(edit, offset, 1, 1, 2)
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(form_box)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        self.tabs.addTab(page, "设置")

    def populate_runtime_environment_combo(self) -> None:
        environments = self.detect_runtime_environments()
        configured_python = self.config.voice_python.strip()
        current_python = str(Path(configured_python).expanduser()) if configured_python else ""
        current_key = self.normalized_path_key(current_python)
        self.runtime_env_combo.blockSignals(True)
        self.runtime_env_combo.clear()
        for label, python_path in environments:
            self.runtime_env_combo.addItem(label, python_path)
        current_index = -1
        for index in range(self.runtime_env_combo.count()):
            if self.normalized_path_key(str(self.runtime_env_combo.itemData(index))) == current_key:
                current_index = index
                break
        current_python_exists = bool(current_python) and Path(current_python).expanduser().exists()
        if current_index < 0 and current_python_exists:
            label = self.runtime_environment_label(Path(current_python))
            self.runtime_env_combo.addItem(label, current_python)
            current_index = self.runtime_env_combo.count() - 1
        elif current_index < 0 and self.runtime_env_combo.count() == 0 and current_python:
            label = self.runtime_environment_label(Path(current_python))
            self.runtime_env_combo.addItem(label, current_python)
            current_index = 0
        self.runtime_env_combo.setCurrentIndex(current_index if current_index >= 0 else 0)
        self.runtime_env_combo.blockSignals(False)
        self.handle_runtime_environment_changed()

    def detect_runtime_environments(self) -> list[tuple[str, str]]:
        env_dirs: list[Path] = []
        conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
        if conda_prefix:
            env_dirs.append(Path(conda_prefix))
        current_executable = Path(sys.executable)
        if current_executable.name.lower() in {"python.exe", "python"}:
            env_dirs.append(current_executable.parent)
        user_profile = Path(os.getenv("USERPROFILE", str(Path.home())))
        env_roots = [
            user_profile / ".conda" / "envs",
            user_profile / "anaconda3" / "envs",
            user_profile / "miniconda3" / "envs",
            user_profile / "mambaforge" / "envs",
            user_profile / "miniforge3" / "envs",
        ]
        for root in env_roots:
            if root.exists():
                env_dirs.extend([path for path in root.iterdir() if path.is_dir()])
        configured_python_text = self.config.voice_python.strip()
        configured_python = Path(configured_python_text).expanduser() if configured_python_text else Path()
        if configured_python_text and configured_python.name.lower() in {"python.exe", "python"}:
            env_dirs.append(configured_python.parent)
        environments: dict[str, tuple[str, str]] = {}
        for env_dir in env_dirs:
            python_path = self.python_path_for_env(env_dir)
            if not python_path.exists():
                continue
            key = self.normalized_path_key(str(python_path))
            environments[key] = (self.runtime_environment_label(python_path), str(python_path))
        return sorted(environments.values(), key=lambda item: item[0].lower())

    def python_path_for_env(self, env_dir: Path) -> Path:
        if os.name == "nt":
            return env_dir / "python.exe"
        return env_dir / "bin" / "python"

    def runtime_environment_label(self, python_path: Path) -> str:
        env_dir = python_path.parent
        env_name = env_dir.name
        return f"{env_name} ({python_path})"

    def normalized_path_key(self, path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve()).lower()
        except Exception:
            return str(Path(path).expanduser()).lower()

    def handle_runtime_environment_changed(self) -> None:
        python_path = self.runtime_env_combo.currentData()
        if python_path:
            self.voice_python_edit.setText(str(python_path))
            self.config.voice_python = str(python_path)

    def selected_runtime_python(self) -> str:
        python_path = self.runtime_env_combo.currentData()
        return str(python_path or self.voice_python_edit.text()).strip()

    def select_runtime_environment_dir(self) -> None:
        selected_python = self.selected_runtime_python()
        current_python = Path(selected_python).expanduser() if selected_python else Path()
        start_dir = str(current_python.parent if current_python.parent.exists() else PROJECT_ROOT)
        env_dir_text = QFileDialog.getExistingDirectory(self, "选择运行环境目录", start_dir)
        if not env_dir_text:
            return
        env_dir = Path(env_dir_text).expanduser()
        python_path = self.python_path_for_env(env_dir)
        if not python_path.exists():
            QMessageBox.warning(self, "运行环境", f"该目录下未找到 Python: {python_path}")
            return
        key = self.normalized_path_key(str(python_path))
        for index in range(self.runtime_env_combo.count()):
            if self.normalized_path_key(str(self.runtime_env_combo.itemData(index))) == key:
                self.runtime_env_combo.setCurrentIndex(index)
                QMessageBox.information(self, "运行环境", f"已找到 Python: {python_path}")
                return
        label = self.runtime_environment_label(python_path)
        self.runtime_env_combo.addItem(label, str(python_path))
        self.runtime_env_combo.setCurrentIndex(self.runtime_env_combo.count() - 1)
        QMessageBox.information(self, "运行环境", f"已找到 Python: {python_path}")

    def create_model_combo(self, options: tuple[str, ...], current_value: str) -> QComboBox:
        combo = QComboBox()
        for option in options:
            combo.addItem(option)
        current_value = current_value.strip()
        if current_value and combo.findText(current_value) < 0:
            combo.addItem(current_value)
        if current_value:
            index = combo.findText(current_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _build_output_overview_tab(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        top = QHBoxLayout()
        self.output_role_combo = QComboBox()
        self.output_role_combo.currentTextChanged.connect(lambda _text: self.load_output_overview())
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(lambda _checked: self.load_output_overview())
        delete_btn = QPushButton("删除合成语音")
        delete_btn.clicked.connect(self.delete_selected_tts_run)
        open_dir_btn = QPushButton("打开输出文件夹")
        open_dir_btn.clicked.connect(self.open_output_overview_dir)
        top.addWidget(QLabel("角色"))
        top.addWidget(self.output_role_combo)
        top.addWidget(refresh_btn)
        top.addWidget(delete_btn)
        top.addWidget(open_dir_btn)
        top.addStretch(1)

        self.output_overview_table = QTableWidget()
        self.output_overview_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.output_overview_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.output_overview_table.itemSelectionChanged.connect(
            lambda: self.handle_table_selection_changed(self.output_overview_table, self.output_overview_rows)
        )
        self.configure_table_editing(self.output_overview_table, editable=False)

        root.addLayout(top)
        root.addWidget(self.output_overview_table, 1)
        self.tabs.addTab(page, "输出音频概览")

    def _wrap_table(self, title: str, table: QTableWidget, buttons: list[tuple[str, Callable[[], None]]]) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        button_row = QHBoxLayout()
        button_row.addWidget(QLabel(title))
        for label, handler in buttons:
            button = QPushButton(label)
            button.clicked.connect(handler)
            button_row.addWidget(button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addWidget(table)
        return widget

    def configure_table_editing(self, table: QTableWidget, *, editable: bool) -> None:
        table.setProperty("editable_table", editable)
        if editable:
            table.setEditTriggers(
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
                | QAbstractItemView.EditTrigger.AnyKeyPressed
            )
        else:
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def create_task_log_view(self) -> QTextEdit:
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setMaximumHeight(130)
        return log_view

    def clear_log_view(self, log_view: QTextEdit | None) -> None:
        if log_view is not None:
            log_view.clear()

    def clear_all_logs(self) -> None:
        for log_view in [self.log_view, self.filter_log_view, self.asr_log_view, self.emotion_log_view]:
            self.clear_log_view(log_view)
        self.process_output = ""
        self.active_log_view = None

    def clear_filter_preview(self) -> None:
        self.manifest_rows = []
        self.filter_results_dirty = False
        self.populate_filter_tables()

    def clear_asr_preview(self) -> None:
        self.asr_rows = []
        self.asr_display_headers = [header for header in self.asr_headers() if header not in self.asr_hidden_headers]
        self.populate_simple_table(self.asr_table, self.asr_rows, self.asr_display_headers)

    def clear_emotion_preview(self) -> None:
        self.emotion_rows = []
        self.emotion_display_headers = [header for header in self.emotion_headers() if header not in self.emotion_hidden_headers]
        self.populate_simple_table(self.emotion_table, self.emotion_rows, self.emotion_display_headers)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.save_settings_without_dialog()
        self.stop_player()
        self.clear_all_logs()
        super().closeEvent(event)

    def select_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", target.text())
        if path:
            target.setText(path)

    def select_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", target.text())
        if path:
            target.setText(path)

    def reload_project_state(self) -> None:
        try:
            data = get_project_state()
        except Exception as exc:
            QMessageBox.critical(self, "项目状态", str(exc))
            return
        self.role_names = [row["role"] for row in data.get("roles", [])]
        self._refresh_role_combos()

    def _refresh_role_combos(self) -> None:
        combos = [
            self.input_role_combo,
            self.filter_role_combo,
            self.asr_role_combo,
            self.emotion_role_combo,
            self.role_combo,
            self.output_role_combo,
        ]
        for combo in combos:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.role_names)
            if current:
                index = combo.findText(current)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def save_settings(self) -> None:
        self.save_settings_without_dialog()
        QMessageBox.information(self, "设置", "设置已保存。")

    def save_settings_without_dialog(self) -> None:
        self.config.voice_python = self.selected_runtime_python()
        self.config.runtime_environment = self.runtime_env_combo.currentText().strip()
        self.config.voxcpm2_model_path = self.voxcpm_path_edit.text().strip()
        self.config.dashscope_api_key = self.api_key_edit.text().strip()
        self.config.deepseek_api_key = self.deepseek_api_key_edit.text().strip()
        self.config.last_backend = self.backend_combo.currentData()
        self.config.synthesis_language = self.synthesis_language_combo.currentData()
        self.config.asr_source_language = self.asr_language_combo.currentData()
        self.config.reference_window = self.top_k_spin.value()
        self.config.gap_ms = self.gap_spin.value()
        self.config.voxcpm2_diffusion_steps = self.voxcpm2_diffusion_steps_spin.value()
        self.config.voxcpm2_cfg_value = self.voxcpm2_cfg_spin.value()
        self.config.asr_model = self.asr_model_edit.currentText().strip()
        self.config.emotion_model = self.emotion_model_edit.currentText().strip()
        self.config.retrieval_rerank_model = self.retrieval_rerank_model_edit.currentText().strip()
        self.config.retrieval_text_model = self.retrieval_text_model_edit.currentText().strip()
        self.config.voice_enrollment_model = self.voice_enrollment_model_edit.currentText().strip()
        self.config.tts_target_model = self.tts_target_model_edit.currentText().strip()
        self.config.save()

    def migrate_workspace(self) -> None:
        self.start_process(["-m", "GUI.backend.workflow_cli", "migrate-workspace"], self.handle_migrate_result)

    def sync_input_audio(self) -> None:
        role = self.input_role_combo.currentText()
        self.start_process(["-m", "GUI.backend.workflow_cli", "sync-input-audio", "--role", role], self.handle_manifest_result)

    def load_manifest_for_input(self) -> None:
        role = self.input_role_combo.currentText()
        self.start_process(["-m", "GUI.backend.workflow_cli", "load-manifest", "--role", role], self.handle_manifest_result)

    def load_processing_state(self) -> None:
        role = self.filter_role_combo.currentText()
        self.filter_results_dirty = False
        self.start_process(["-m", "GUI.backend.workflow_cli", "load-manifest", "--role", role], self.handle_processing_manifest_result)

    def load_duration_stats(self) -> None:
        role = self.input_role_combo.currentText()
        self.start_process(["-m", "GUI.backend.workflow_cli", "duration-stats", "--role", role], self.handle_duration_stats)

    def save_reference_selection(self) -> None:
        if not self.manifest_rows:
            QMessageBox.warning(self, "输入音频", "没有可保存的音频。")
            return
        role = self.input_role_combo.currentText()
        sha_values = self.checked_reference_sha_values()
        json_path = self.write_runtime_json("reference_selection.json", sha_values)
        args = ["-m", "GUI.backend.workflow_cli", "save-reference-selection", "--role", role, "--input-json", str(json_path)]
        self.start_process(args, self.handle_manifest_result)

    def update_selected_manifest_status(self, status: str) -> None:
        sha_values = self.selected_sha_values(self.input_table, self.manifest_rows)
        if not sha_values:
            QMessageBox.warning(self, "输入音频", "请先选择音频。")
            return
        self.update_status(self.input_role_combo.currentText(), sha_values, status, self.handle_manifest_result)

    def move_filter_status(self, table: QTableWidget, rows: list[dict[str, Any]], status: str) -> None:
        sha_values = self.selected_sha_values(table, rows)
        if not sha_values:
            QMessageBox.warning(self, "筛选音频", "请先选择音频。")
            return
        wanted = set(sha_values)
        for row in self.manifest_rows:
            if row.get("sha256") in wanted:
                row["filter_status"] = status
        self.filter_results_dirty = True
        self.populate_filter_tables()

    def update_status(self, role: str, sha_values: list[str], status: str, callback: Callable[[dict[str, Any]], None]) -> None:
        json_path = self.write_runtime_json("status_selection.json", sha_values)
        args = ["-m", "GUI.backend.workflow_cli", "update-filter-status", "--role", role, "--status", status, "--input-json", str(json_path)]
        self.start_process(args, callback)

    def run_filter_audio(self) -> None:
        role = self.filter_role_combo.currentText()
        args = [
            "-m", "GUI.backend.workflow_cli", "filter-audio",
            "--role", role,
            "--confirm-threshold", str(self.confirm_threshold_spin.value()),
            "--review-threshold", str(self.review_threshold_spin.value()),
        ]
        self.filter_log_view.clear()
        self.start_process(args, self.handle_filter_result, log_view=self.filter_log_view, process_kind="filter")

    def save_filter_results(self) -> None:
        if not self.manifest_rows:
            QMessageBox.warning(self, "音频筛选", "没有可保存的筛选结果。")
            return
        role = self.filter_role_combo.currentText()
        json_path = self.write_runtime_json("filter_rows.json", self.manifest_rows)
        self.start_process(
            ["-m", "GUI.backend.workflow_cli", "save-filter-results", "--role", role, "--input-json", str(json_path)],
            self.handle_save_filter_result,
            log_view=self.filter_log_view,
        )

    def load_asr(self) -> None:
        role = self.asr_role_combo.currentText()
        self.start_process(["-m", "GUI.backend.workflow_cli", "load-asr", "--role", role], self.handle_asr_result)

    def run_asr(self) -> None:
        self.save_settings_without_dialog()
        role = self.asr_role_combo.currentText()
        if not self.asr_model_edit.currentText().strip():
            QMessageBox.warning(self, "语音识别", "ASR 模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-asr",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--model", self.asr_model_edit.currentText().strip(),
            "--language", self.asr_language_combo.currentData(),
        ]
        self.asr_log_view.clear()
        self.start_process(args, self.handle_asr_result, log_view=self.asr_log_view, process_kind="asr")

    def run_single_asr(self) -> None:
        self.save_settings_without_dialog()
        row = self.selected_row(self.asr_table, self.asr_rows)
        if not row:
            QMessageBox.warning(self, "语音识别", "请先选择要识别的音频。")
            return
        sha256 = str(row.get("sha256", "") or "").strip()
        if not sha256:
            QMessageBox.warning(self, "语音识别", "所选音频缺少 sha256，无法运行单条 ASR。")
            return
        role = self.asr_role_combo.currentText()
        if not self.asr_model_edit.currentText().strip():
            QMessageBox.warning(self, "语音识别", "ASR 模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-asr",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--model", self.asr_model_edit.currentText().strip(),
            "--language", self.asr_language_combo.currentData(),
            "--sha256", sha256,
            "--force",
        ]
        self.asr_log_view.clear()
        self.start_process(args, self.handle_asr_result, log_view=self.asr_log_view, process_kind="asr")

    def run_translation(self) -> None:
        self.save_settings_without_dialog()
        if not self.asr_rows:
            QMessageBox.warning(self, "语音识别", "请先加载或运行 ASR。")
            return
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "语音识别", "翻译模型不能为空。")
            return
        if text_model.startswith("deepseek") and not self.deepseek_api_key_edit.text().strip():
            QMessageBox.warning(self, "语音识别", "当前翻译模型需要 Deepseek API Key。")
            return
        if not text_model.startswith("deepseek") and not self.api_key_edit.text().strip():
            QMessageBox.warning(self, "语音识别", "当前翻译模型需要 Qwen API Key。")
            return
        role = self.asr_role_combo.currentText()
        rows = self.collect_hidden_table_rows(self.asr_table, self.asr_rows, self.asr_display_headers)
        json_path = self.write_runtime_json("asr_translation_rows.json", rows)
        args = [
            "-m", "GUI.backend.workflow_cli", "run-translation",
            "--role", role,
            "--input-json", str(json_path),
            "--language", self.asr_language_combo.currentData(),
            "--model", text_model,
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
        ]
        self.asr_log_view.clear()
        self.start_process(args, self.handle_asr_result, log_view=self.asr_log_view, process_kind="asr")

    def run_single_translation(self) -> None:
        self.save_settings_without_dialog()
        row = self.selected_row(self.asr_table, self.asr_rows)
        if not row:
            QMessageBox.warning(self, "语音识别", "请先选择要翻译的音频。")
            return
        sha256 = str(row.get("sha256", "") or "").strip()
        if not sha256:
            QMessageBox.warning(self, "语音识别", "所选音频缺少 sha256，无法运行单条翻译。")
            return
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "语音识别", "翻译模型不能为空。")
            return
        if text_model.startswith("deepseek") and not self.deepseek_api_key_edit.text().strip():
            QMessageBox.warning(self, "语音识别", "当前翻译模型需要 Deepseek API Key。")
            return
        if not text_model.startswith("deepseek") and not self.api_key_edit.text().strip():
            QMessageBox.warning(self, "语音识别", "当前翻译模型需要 Qwen API Key。")
            return
        role = self.asr_role_combo.currentText()
        rows = self.collect_hidden_table_rows(self.asr_table, self.asr_rows, self.asr_display_headers)
        json_path = self.write_runtime_json("asr_translation_rows.json", rows)
        args = [
            "-m", "GUI.backend.workflow_cli", "run-translation",
            "--role", role,
            "--input-json", str(json_path),
            "--language", self.asr_language_combo.currentData(),
            "--model", text_model,
            "--sha256", sha256,
            "--force",
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
        ]
        self.asr_log_view.clear()
        self.start_process(args, self.handle_asr_result, log_view=self.asr_log_view, process_kind="asr")

    def save_asr(self) -> None:
        role = self.asr_role_combo.currentText()
        rows = self.collect_hidden_table_rows(self.asr_table, self.asr_rows, self.asr_display_headers)
        json_path = self.write_runtime_json("asr_rows.json", rows)
        self.start_process(["-m", "GUI.backend.workflow_cli", "save-asr", "--role", role, "--input-json", str(json_path)], self.handle_save_generic)

    def load_emotion(self) -> None:
        role = self.emotion_role_combo.currentText()
        self.start_process(["-m", "GUI.backend.workflow_cli", "load-emotion", "--role", role], self.handle_emotion_result)

    def run_emotion_description(self) -> None:
        role = self.emotion_role_combo.currentText()
        if not self.emotion_model_edit.currentText().strip():
            QMessageBox.warning(self, "情感标定", "情感标定模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-emotion-description",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--model", self.emotion_model_edit.currentText().strip(),
        ]
        self.emotion_log_view.clear()
        self.start_process(args, self.handle_emotion_result, log_view=self.emotion_log_view, process_kind="emotion")

    def run_single_emotion_description(self) -> None:
        row = self.selected_row(self.emotion_table, self.emotion_rows)
        if not row:
            QMessageBox.warning(self, "情感标定", "请先选择要标定的音频。")
            return
        sha256 = str(row.get("sha256", "") or "").strip()
        if not sha256:
            QMessageBox.warning(self, "情感标定", "所选音频缺少 sha256，无法运行情感描述。")
            return
        if not str(row.get("语音文本", "") or "").strip():
            QMessageBox.warning(self, "情感标定", "该音频尚未完成 ASR，无法运行情感描述。")
            return
        role = self.emotion_role_combo.currentText()
        if not self.emotion_model_edit.currentText().strip():
            QMessageBox.warning(self, "情感标定", "情感标定模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-emotion-description",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--model", self.emotion_model_edit.currentText().strip(),
            "--sha256", sha256,
            "--force",
        ]
        self.emotion_log_view.clear()
        self.start_process(args, self.handle_emotion_result, log_view=self.emotion_log_view, process_kind="emotion")

    def run_emotion_keywords(self) -> None:
        role = self.emotion_role_combo.currentText()
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "情感标定", "关键词提取模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-emotion-keywords",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
            "--text-model", text_model,
        ]
        self.emotion_log_view.clear()
        self.start_process(args, self.handle_emotion_result, log_view=self.emotion_log_view, process_kind="emotion")

    def run_single_emotion_keywords(self) -> None:
        row = self.selected_row(self.emotion_table, self.emotion_rows)
        if not row:
            QMessageBox.warning(self, "情感标定", "请先选择要提取关键词的音频。")
            return
        sha256 = str(row.get("sha256", "") or "").strip()
        if not sha256:
            QMessageBox.warning(self, "情感标定", "所选音频缺少 sha256，无法提取关键词。")
            return
        if not str(row.get("自然语言描述", "") or "").strip():
            QMessageBox.warning(self, "情感标定", "该音频尚未完成情感描述，无法提取关键词。")
            return
        role = self.emotion_role_combo.currentText()
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "情感标定", "关键词提取模型不能为空。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "run-emotion-keywords",
            "--role", role,
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
            "--text-model", text_model,
            "--sha256", sha256,
            "--force",
        ]
        self.emotion_log_view.clear()
        self.start_process(args, self.handle_emotion_result, log_view=self.emotion_log_view, process_kind="emotion")

    def save_emotion(self) -> None:
        role = self.emotion_role_combo.currentText()
        rows = self.collect_hidden_table_rows(self.emotion_table, self.emotion_rows, self.emotion_display_headers)
        json_path = self.write_runtime_json("emotion_rows.json", rows)
        self.start_process(["-m", "GUI.backend.workflow_cli", "save-emotion", "--role", role, "--input-json", str(json_path)], self.handle_save_generic)

    def load_output_overview(self, silent: bool = False) -> None:
        role = self.output_role_combo.currentText().strip()
        if not role:
            self.output_overview_rows = []
            self.populate_simple_table(self.output_overview_table, [], self.output_overview_display_headers)
            return
        self.start_process(["-m", "GUI.backend.workflow_cli", "list-tts-runs", "--role", role], self.handle_output_overview_result, silent=silent)

    def delete_selected_tts_run(self) -> None:
        row = self.selected_row(self.output_overview_table, self.output_overview_rows)
        if not row:
            QMessageBox.warning(self, "输出音频概览", "请先选择要删除的合成语音。")
            return
        target_audio_path = str(row.get("audio_path", "") or "").strip()
        run_summary_path = str(row.get("run_summary_path", "") or "").strip()
        audio_name = str(row.get("语音名称", "") or "").strip() or "所选合成语音"
        if not run_summary_path:
            QMessageBox.warning(self, "输出音频概览", "所选记录缺少 run_summary 路径，无法删除。")
            return
        answer = QMessageBox.question(
            self,
            "删除合成语音",
            f"确认删除 {audio_name} 及其配套文件吗？该操作会物理删除文件，且不可恢复。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if target_audio_path and self.current_audio_path:
            current_audio = str(Path(self.current_audio_path).resolve())
            target_audio = str(Path(target_audio_path).resolve())
            if current_audio == target_audio:
                self.clear_player_state()
                QApplication.processEvents()
        role = self.output_role_combo.currentText().strip()
        self.start_process(
            ["-m", "GUI.backend.workflow_cli", "delete-tts-run", "--role", role, "--run-summary-path", run_summary_path],
            self.handle_delete_tts_run_result,
        )

    def run_retrieval(self) -> None:
        self.save_settings_without_dialog()
        if not self.validate_tts_inputs(require_backend=False):
            return
        self.log_view.clear()
        self.result_path_edit.clear()
        self.retrieval_rows = []
        self.last_retrieval_result_path = ""
        self.last_session_dir = ""
        self.populate_retrieval_table()
        args = [
            "-m", "GUI.backend.workflow_cli", "retrieve",
            "--role", self.role_combo.currentText(),
            "--transcript-text", self.transcript_edit.toPlainText().strip(),
            "--voice-description", self.retrieval_guidance_edit.toPlainText().strip(),
            "--api-key", self.api_key_edit.text().strip(),
            "--rerank-model", self.retrieval_rerank_model_edit.currentText().strip(),
            "--text-model", self.retrieval_text_model_edit.currentText().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
            "--reference-window", str(self.top_k_spin.value()),
        ]
        self.start_process(args, self.handle_retrieval_result, log_view=self.log_view)

    def synthesize_tts(self) -> None:
        self.save_settings_without_dialog()
        if not self.validate_tts_inputs(require_backend=True):
            return
        if not self.last_retrieval_result_path or not Path(self.last_retrieval_result_path).exists():
            QMessageBox.warning(self, "配音", "请先完成检索。")
            return
        if not self.last_session_dir:
            QMessageBox.warning(self, "配音", "检索会话目录无效，请重新检索。")
            return
        timbre_rows = self.selected_timbre_rows()
        if not timbre_rows:
            QMessageBox.warning(self, "配音", "请至少勾选一条音色参考音频。")
            return
        selected_results_path = self.write_runtime_json("retrieval_selection.json", timbre_rows)
        style_rows = self.selected_style_rows()
        style_results_path = self.write_runtime_json("style_selection.json", style_rows) if style_rows else ""
        backend = self.backend_combo.currentData()
        if backend == "voxcpm2_local_hifi" and not style_rows:
            QMessageBox.warning(self, "配音", "Hifi 模式建议至少勾选一条语气参考音频以提供说话风格参考。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "synthesize",
            "--role", self.role_combo.currentText(),
            "--transcript-text", self.transcript_edit.toPlainText().strip(),
            "--voice-description", self.retrieval_guidance_edit.toPlainText().strip(),
            "--synthesis-guidance", self.synthesis_guidance_edit.toPlainText().strip(),
            "--backend", backend,
            "--synthesis-language", self.synthesis_language_combo.currentData(),
            "--reference-window", str(self.top_k_spin.value()),
            "--reference-min-seconds", str(self.config.reference_min_seconds),
            "--gap-ms", str(self.gap_spin.value()),
            "--voxcpm2-diffusion-steps", str(self.voxcpm2_diffusion_steps_spin.value()),
            "--voxcpm2-cfg-value", str(self.voxcpm2_cfg_spin.value()),
            "--voxcpm2-model-path", self.voxcpm_path_edit.text().strip(),
            "--retrieval-result-path", self.last_retrieval_result_path,
            "--session-dir", self.last_session_dir,
            "--selected-results-json", str(selected_results_path),
            "--style-results-json", str(style_results_path) if style_results_path else "",
            "--api-key", self.api_key_edit.text().strip(),
            "--voice-enrollment-model", self.voice_enrollment_model_edit.currentText().strip(),
            "--tts-target-model", self.tts_target_model_edit.currentText().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
        ]
        self.start_process(args, self.handle_tts_result, log_view=self.log_view)

    def validate_tts_inputs(self, *, require_backend: bool) -> bool:
        if not self.role_combo.currentText().strip():
            QMessageBox.warning(self, "输入检查", "请选择角色。")
            return False
        if not self.transcript_edit.toPlainText().strip():
            QMessageBox.warning(self, "输入检查", "配音台词不能为空。")
            return False
        if not self.retrieval_guidance_edit.toPlainText().strip():
            QMessageBox.warning(self, "输入检查", "检索声音指导不能为空。")
            return False
        voice_python = Path(self.config.voice_python).expanduser() if self.config.voice_python.strip() else Path()
        if not self.config.voice_python.strip() or not voice_python.is_file():
            QMessageBox.warning(self, "输入检查", f"voice Python 不存在: {self.config.voice_python}")
            return False
        if require_backend and self.backend_combo.currentData() != "api" and not Path(self.voxcpm_path_edit.text().strip()).exists():
            QMessageBox.warning(self, "输入检查", f"VoxCPM2 模型目录不存在: {self.voxcpm_path_edit.text().strip()}")
            return False
        if not self.api_key_edit.text().strip():
            QMessageBox.warning(self, "输入检查", "检索流程需要 Qwen API Key，请先在设置页填写。")
            return False
        if self.retrieval_text_model_edit.currentText().strip().startswith("deepseek") and not self.deepseek_api_key_edit.text().strip():
            QMessageBox.warning(self, "输入检查", "Deepseek 最终排序模型需要 Deepseek API Key，请先在设置页填写。")
            return False
        required_models = [
            ("检索 rerank 模型", self.retrieval_rerank_model_edit.currentText().strip()),
            ("最终排序文本模型", self.retrieval_text_model_edit.currentText().strip()),
        ]
        if require_backend and self.backend_combo.currentData() == "api":
            required_models.extend(
                [
                    ("声音复刻模型", self.voice_enrollment_model_edit.currentText().strip()),
                    ("云端语音合成模型", self.tts_target_model_edit.currentText().strip()),
                ]
            )
        for label, value in required_models:
            if not value:
                QMessageBox.warning(self, "输入检查", f"{label}不能为空。")
                return False
        return True

    def start_process(
        self,
        args: list[str],
        callback: Callable[[dict[str, Any]], None],
        log_view: QTextEdit | None = None,
        process_kind: str = "",
        silent: bool = False,
    ) -> None:
        if self.current_process is not None:
            QMessageBox.warning(self, "任务执行", "已有任务正在执行，请等待结束。")
            return
        self.process_output = ""
        self.stop_requested = False
        self.current_process_kind = process_kind
        self.active_log_view = log_view or self.log_view
        process = QProcess(self)
        self.current_process = process
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        if self.api_key_edit.text().strip():
            env.insert("DASHSCOPE_API_KEY", self.api_key_edit.text().strip())
        if self.deepseek_api_key_edit.text().strip():
            env.insert("DEEPSEEK_API_KEY", self.deepseek_api_key_edit.text().strip())
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(lambda: self._append_process_output(process.readAllStandardOutput().data().decode("utf-8", errors="replace")))
        process.readyReadStandardError.connect(lambda: self._append_process_output(process.readAllStandardError().data().decode("utf-8", errors="replace")))
        process.finished.connect(lambda _code, _status: self._process_finished(callback))
        if not silent:
            self.append_log("开始运行")
            self.append_log(f"> {self.config.voice_python} {' '.join(args)}")
        process.start(self.config.voice_python, args)

    def stop_process(self, process_kind: str, title: str, clear_preview: Callable[[], None]) -> None:
        process = self.current_process
        if process is None or self.current_process_kind != process_kind:
            QMessageBox.warning(self, title, f"当前没有正在运行的{title}任务。")
            return
        self.stop_requested = True
        self.append_log(f"{title}任务已请求停止。")
        clear_preview()
        process.terminate()
        QTimer.singleShot(3000, lambda: self.force_kill_process(process))

    def force_kill_process(self, process: QProcess) -> None:
        try:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
        except RuntimeError:
            return

    def _append_process_output(self, text: str) -> None:
        self.process_output += text
        for line in text.splitlines():
            if not line.startswith(RESULT_PREFIX):
                self.append_log(line)

    def _process_finished(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if self.stop_requested:
            self.append_log("任务已停止，预览结果已清空。")
            self.current_process = None
            self.current_process_kind = ""
            self.stop_requested = False
            self.active_log_view = None
            self.process_output = ""
            return
        result = self.extract_result()
        self.current_process = None
        self.current_process_kind = ""
        self.active_log_view = None
        if result is None:
            QMessageBox.critical(self, "任务执行", "任务结束但未返回结构化结果，请查看日志。")
            return
        self.last_result = result
        if not result.get("success"):
            QMessageBox.critical(self, "任务失败", result.get("message", "未知错误"))
            return
        callback(result)
        self.reload_project_state()

    def extract_result(self) -> dict[str, Any] | None:
        for line in reversed(self.process_output.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX) :])
        return None

    def handle_migrate_result(self, result: dict[str, Any]) -> None:
        QMessageBox.information(self, "迁移", result.get("message", "迁移完成。"))
        self.reload_project_state()

    def handle_manifest_result(self, result: dict[str, Any]) -> None:
        self.manifest_rows = result.get("data", {}).get("rows", [])
        self.populate_manifest_table(self.input_table, self.manifest_rows)

    def handle_processing_manifest_result(self, result: dict[str, Any]) -> None:
        self.manifest_rows = result.get("data", {}).get("rows", [])
        self.populate_filter_tables()

    def handle_duration_stats(self, result: dict[str, Any]) -> None:
        rows = result.get("data", {}).get("buckets", [])
        self.populate_simple_table(self.duration_table, rows, ["bucket", "count"])

    def handle_filter_result(self, result: dict[str, Any]) -> None:
        self.manifest_rows = result.get("data", {}).get("rows", [])
        self.filter_results_dirty = True
        self.populate_filter_tables()
        QMessageBox.information(self, "筛选", result.get("message", "筛选完成，请确认后保存。"))

    def handle_save_filter_result(self, result: dict[str, Any]) -> None:
        self.manifest_rows = result.get("data", {}).get("rows", [])
        self.filter_results_dirty = False
        self.populate_filter_tables()
        QMessageBox.information(self, "音频筛选", result.get("message", "筛选结果已保存。"))

    def handle_asr_result(self, result: dict[str, Any]) -> None:
        self.asr_rows = result.get("data", {}).get("rows", [])
        self.asr_display_headers = [header for header in self.asr_headers() if header not in self.asr_hidden_headers]
        self.populate_simple_table(self.asr_table, self.asr_rows, self.asr_display_headers)

    def handle_emotion_result(self, result: dict[str, Any]) -> None:
        self.emotion_rows = result.get("data", {}).get("rows", [])
        self.emotion_display_headers = [header for header in self.emotion_headers() if header not in self.emotion_hidden_headers]
        self.populate_simple_table(self.emotion_table, self.emotion_rows, self.emotion_display_headers)

    def handle_output_overview_result(self, result: dict[str, Any]) -> None:
        self.output_overview_rows = result.get("data", {}).get("rows", [])
        self.populate_simple_table(self.output_overview_table, self.output_overview_rows, self.output_overview_display_headers)

    def handle_delete_tts_run_result(self, result: dict[str, Any]) -> None:
        data = result.get("data", {})
        self.output_overview_rows = data.get("rows", [])
        self.populate_simple_table(self.output_overview_table, self.output_overview_rows, self.output_overview_display_headers)
        deleted_paths = {
            str(Path(path).resolve())
            for path in data.get("deleted_paths", [])
            if str(path).strip()
        }
        deleted_audio_path = str(data.get("deleted_audio_path", "") or "").strip()
        deleted_session_dir = str(data.get("deleted_session_dir", "") or "").strip()
        session_cleared = bool(data.get("session_cleared"))
        current_audio_resolved = str(Path(self.current_audio_path).resolve()) if self.current_audio_path else ""
        deleted_audio_resolved = str(Path(deleted_audio_path).resolve()) if deleted_audio_path else ""

        if deleted_audio_resolved and current_audio_resolved == deleted_audio_resolved:
            self.clear_player_state()
        elif current_audio_resolved and current_audio_resolved in deleted_paths:
            self.clear_player_state()
        if deleted_audio_resolved and self.result_path_edit.text().strip():
            current_result_path = str(Path(self.result_path_edit.text().strip()).resolve())
            if current_result_path == deleted_audio_resolved:
                self.result_path_edit.clear()
        if session_cleared and deleted_session_dir and self.last_session_dir and Path(deleted_session_dir).resolve() == Path(self.last_session_dir).resolve():
            self.last_session_dir = ""
            self.last_retrieval_result_path = ""

        QMessageBox.information(self, "输出音频概览", result.get("message", "合成语音已删除。"))

    def handle_save_generic(self, result: dict[str, Any]) -> None:
        QMessageBox.information(self, "保存", result.get("message", "保存完成。"))

    def handle_tts_result(self, result: dict[str, Any]) -> None:
        audio_path = result.get("output_paths", {}).get("synthesized_audio_path", "")
        self.result_path_edit.setText(audio_path)
        if audio_path and Path(audio_path).exists():
            self.load_audio_into_player(audio_path, autoplay=False)
        if self.output_role_combo.currentText().strip() == self.role_combo.currentText().strip():
            self.output_overview_rows = result.get("data", {}).get("rows", self.output_overview_rows)
            self.populate_simple_table(self.output_overview_table, self.output_overview_rows, self.output_overview_display_headers)
        QMessageBox.information(self, "配音", result.get("message", "配音生成完成。"))

    def handle_retrieval_result(self, result: dict[str, Any]) -> None:
        self.retrieval_rows = result.get("data", {}).get("top_results", [])
        self.last_retrieval_result_path = result.get("output_paths", {}).get("retrieval_result_path", "")
        self.last_session_dir = result.get("output_paths", {}).get("session_dir", "")
        self.populate_retrieval_table()
        QMessageBox.information(self, "检索", result.get("message", "检索完成。"))

    def run_auto_guidance(self) -> None:
        self.save_settings_without_dialog()
        transcript = self.transcript_edit.toPlainText().strip()
        if not transcript:
            QMessageBox.warning(self, "自动指导", "请先在「配音台词」框中输入台词。")
            return
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "自动指导", "请先在设置页配置文本模型。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "auto-guidance",
            "--role", self.role_combo.currentText(),
            "--transcript-text", transcript,
            "--text-model", text_model,
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
        ]
        self.start_process(args, self.handle_auto_guidance_result, log_view=self.log_view)

    def handle_auto_guidance_result(self, result: dict[str, Any]) -> None:
        guidance = result.get("data", {}).get("guidance", "")
        self.retrieval_guidance_edit.setPlainText(guidance)
        QMessageBox.information(self, "自动指导", result.get("message", "自动指导生成完成。"))

    def run_sentence_split(self) -> None:
        self.save_settings_without_dialog()
        text = self.lt_input_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "自动分句", "请先输入要分句的文本。")
            return
        text_model = self.retrieval_text_model_edit.currentText().strip()
        if not text_model:
            QMessageBox.warning(self, "自动分句", "请先在设置页配置文本模型。")
            return
        args = [
            "-m", "GUI.backend.workflow_cli", "split-sentences",
            "--role", self.role_combo.currentText(),
            "--transcript-text", text,
            "--text-model", text_model,
            "--api-key", self.api_key_edit.text().strip(),
            "--deepseek-api-key", self.deepseek_api_key_edit.text().strip(),
        ]
        if not self.lt_thinking_check.isChecked():
            args.append("--no-thinking")
        self.start_process(args, self.handle_sentence_split_result, log_view=self.lt_log_view)

    def handle_sentence_split_result(self, result: dict[str, Any]) -> None:
        segments = result.get("data", {}).get("segments", [])
        self.lt_segment_table.setRowCount(len(segments))
        for row_idx, seg in enumerate(segments):
            text_item = QTableWidgetItem(seg.get("text", ""))
            text_item.setFlags(text_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.lt_segment_table.setItem(row_idx, 0, text_item)
            guidance_item = QTableWidgetItem(seg.get("guidance", ""))
            guidance_item.setFlags(guidance_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.lt_segment_table.setItem(row_idx, 1, guidance_item)
        self.lt_segment_table.resizeColumnsToContents()
        QMessageBox.information(self, "自动分句", result.get("message", f"分句完成，共 {len(segments)} 段。"))

    def _jump_segment_to_dubbing(self, row: int, _col: int) -> None:
        text_item = self.lt_segment_table.item(row, 0)
        guidance_item = self.lt_segment_table.item(row, 1)
        transcript = text_item.text().strip() if text_item else ""
        guidance = guidance_item.text().strip() if guidance_item else ""
        if transcript:
            self.transcript_edit.setPlainText(transcript)
            self.retrieval_guidance_edit.setPlainText(guidance)
            self.tabs.setCurrentIndex(4)

    def populate_manifest_table(self, table: QTableWidget, rows: list[dict[str, Any]]) -> None:
        headers = ["file_name", "duration_seconds", "filter_status", "selected_reference"]
        if table is self.input_table:
            self.populate_input_manifest_table(rows, headers)
            return
        self.populate_simple_table(table, rows, headers)

    def populate_input_manifest_table(self, rows: list[dict[str, Any]], headers: list[str]) -> None:
        table = self.input_table
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        reference_column = headers.index("selected_reference")
        centered_headers = {"duration_seconds", "filter_status", "selected_reference"}
        for r, row in enumerate(rows):
            for c, header in enumerate(headers):
                if c == reference_column:
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    checkbox = QCheckBox()
                    checkbox.setChecked(self.is_truthy(row.get("selected_reference")))
                    checkbox_container = QWidget()
                    checkbox_layout = QHBoxLayout(checkbox_container)
                    checkbox_layout.setContentsMargins(0, 0, 0, 0)
                    checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    checkbox_layout.addWidget(checkbox)
                    table.setCellWidget(r, c, checkbox_container)
                else:
                    item = QTableWidgetItem(str(row.get(header, "")))
                    if header in centered_headers:
                        item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def populate_filter_tables(self) -> None:
        self.populate_manifest_table(self.confirm_table, self.confirm_rows())
        self.populate_manifest_table(self.review_table, self.review_rows())

    def populate_simple_table(self, table: QTableWidget, rows: list[dict[str, Any]], headers: list[str]) -> None:
        editable = bool(table.property("editable_table"))
        centered_headers = {"duration_seconds", "filter_status", "selected_reference", "音频时长(秒)"}
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, header in enumerate(headers):
                value = self.reference_display_value(row.get(header)) if header == "selected_reference" else self.table_display_value(row, header)
                item = QTableWidgetItem(value)
                if header in centered_headers:
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def table_display_value(self, row: dict[str, Any], header: str) -> str:
        if header == EMOTION_TONE_KEYWORDS:
            return str(row.get("情绪语气", ""))
        if header == EMOTION_DELIVERY_KEYWORDS:
            return str(row.get("音频表达技巧", ""))
        return str(row.get(header, ""))

    def is_truthy(self, value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "是", "勾选", "checked"}

    def reference_display_value(self, value: Any) -> str:
        return "✓" if self.is_truthy(value) else "✕"

    def populate_retrieval_table(self) -> None:
        headers = ["音色参考", "语气参考", "语音文件", "音频时长(秒)", "语音文本", "情绪语气", "语音技巧"]
        self.retrieval_table.blockSignals(True)
        self.retrieval_table.setColumnCount(len(headers))
        self.retrieval_table.setHorizontalHeaderLabels(headers)
        self.retrieval_table.setRowCount(len(self.retrieval_rows))
        for r, row in enumerate(self.retrieval_rows):
            # Column 0: timbre checkbox
            timbre_item = QTableWidgetItem("")
            timbre_item.setFlags(timbre_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            timbre_checkbox = QCheckBox()
            timbre_checkbox.setChecked(bool(row.get("selected_for_timbre", False)))
            timbre_container = QWidget()
            timbre_layout = QHBoxLayout(timbre_container)
            timbre_layout.setContentsMargins(0, 0, 0, 0)
            timbre_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            timbre_layout.addWidget(timbre_checkbox)
            self.retrieval_table.setItem(r, 0, timbre_item)
            self.retrieval_table.setCellWidget(r, 0, timbre_container)

            # Column 1: style checkbox
            style_item = QTableWidgetItem("")
            style_item.setFlags(style_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            style_checkbox = QCheckBox()
            style_checkbox.setChecked(bool(row.get("selected_for_style", False)))
            style_container = QWidget()
            style_layout = QHBoxLayout(style_container)
            style_layout.setContentsMargins(0, 0, 0, 0)
            style_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            style_layout.addWidget(style_checkbox)
            self.retrieval_table.setItem(r, 1, style_item)
            self.retrieval_table.setCellWidget(r, 1, style_container)

            values = [
                str(row.get("语音文件", "")),
                f"{float(row.get('duration_seconds', 0) or 0):.1f}",
                str(row.get("语音文本", "")),
                str(row.get("情绪语气", "")),
                str(row.get("语音技巧", "")),
            ]
            for c, value in enumerate(values, start=2):
                item = QTableWidgetItem(value)
                if c == 3:
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.retrieval_table.setItem(r, c, item)
        self.retrieval_table.blockSignals(False)
        self.retrieval_table.resizeColumnsToContents()
        self.retrieval_table.setColumnWidth(0, 74)
        self.retrieval_table.setColumnWidth(1, 74)

    def _read_checkbox_column(self, col: int, key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_index, base_row in enumerate(self.retrieval_rows):
            widget = self.retrieval_table.cellWidget(row_index, col)
            checkbox = widget.findChild(QCheckBox) if widget is not None else None
            selected = bool(base_row.get(key, False))
            if checkbox is not None:
                selected = checkbox.isChecked()
            base_row[key] = selected
            if selected:
                rows.append(dict(base_row))
        return rows

    def selected_timbre_rows(self) -> list[dict[str, Any]]:
        return self._read_checkbox_column(0, "selected_for_timbre")

    def selected_style_rows(self) -> list[dict[str, Any]]:
        return self._read_checkbox_column(1, "selected_for_style")

    def selected_reference_duration_seconds(self, rows: list[dict[str, Any]]) -> float:
        total_seconds = sum(float(row.get("duration_seconds", 0) or 0) for row in rows)
        if len(rows) > 1:
            total_seconds += (len(rows) - 1) * float(self.gap_spin.value()) / 1000.0
        return total_seconds

    def collect_hidden_table_rows(self, table: QTableWidget, base_rows: list[dict[str, Any]], display_headers: list[str]) -> list[dict[str, Any]]:
        rows = [dict(row) for row in base_rows]
        for r, row in enumerate(rows):
            for c, header in enumerate(display_headers):
                item = table.item(r, c)
                row[header] = item.text() if item is not None else ""
            if table is self.emotion_table:
                row["情绪语气"] = str(row.get(EMOTION_TONE_KEYWORDS, row.get("情绪语气", "")) or "").strip()
                row["音频表达技巧"] = str(row.get(EMOTION_DELIVERY_KEYWORDS, row.get("音频表达技巧", "")) or "").strip()
                row["关键词"] = "，".join(
                    part
                    for part in [row.get("情绪语气", ""), row.get("音频表达技巧", "")]
                    if str(part).strip()
                )
        return rows

    def collect_table_rows(self, table: QTableWidget, headers: list[str]) -> list[dict[str, str]]:
        rows = []
        for r in range(table.rowCount()):
            row = {}
            for c, header in enumerate(headers):
                item = table.item(r, c)
                row[header] = item.text() if item else ""
            rows.append(row)
        return rows

    def selected_sha_values(self, table: QTableWidget, rows: list[dict[str, Any]]) -> list[str]:
        indexes = sorted({item.row() for item in table.selectedItems()})
        return [str(rows[index].get("sha256", "")) for index in indexes if index < len(rows) and rows[index].get("sha256")]

    def checked_reference_sha_values(self) -> list[str]:
        headers = ["file_name", "duration_seconds", "filter_status", "selected_reference"]
        reference_column = headers.index("selected_reference")
        sha_values: list[str] = []
        for index, row in enumerate(self.manifest_rows):
            widget = self.input_table.cellWidget(index, reference_column)
            checkbox = widget.findChild(QCheckBox) if widget is not None else None
            checked = checkbox is not None and checkbox.isChecked()
            row["selected_reference"] = "true" if checked else "false"
            if checked and row.get("sha256"):
                sha_values.append(str(row["sha256"]))
        return sha_values

    def selected_row(self, table: QTableWidget, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        indexes = sorted({item.row() for item in table.selectedItems()})
        if not indexes or indexes[0] >= len(rows):
            return None
        return rows[indexes[0]]

    def confirm_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.manifest_rows if row.get("filter_status") == "confirmed"]

    def review_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.manifest_rows if row.get("filter_status") == "review"]

    def asr_headers(self) -> list[str]:
        return ["sha256", "role", "file_name", "file_path", "model", "translation_model", "语音文件", "原始语言", "原始文本", "中文文本", "ASR错误", "翻译错误"]

    def emotion_headers(self) -> list[str]:
        return [
            "sha256",
            "model",
            "error",
            "索引",
            "语音文件",
            "语音文本",
            "音频路径",
            "自然语言描述",
            "情绪语气",
            "音频表达技巧",
            "关键词",
            EMOTION_TONE_KEYWORDS,
            EMOTION_DELIVERY_KEYWORDS,
        ]

    def format_duration(self, milliseconds: int) -> str:
        total_seconds = max(0, int(milliseconds // 1000))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def update_player_time_label(self, position: int | None = None, duration: int | None = None) -> None:
        current_position = self.player.position() if position is None else position
        current_duration = self.player.duration() if duration is None else duration
        self.player_time_label.setText(
            f"{self.format_duration(current_position)} / {self.format_duration(current_duration)}"
        )

    def update_player_metadata(self, path: str) -> None:
        audio_path = Path(path)
        self.current_audio_path = str(audio_path)
        self.player_title_label.setText(audio_path.name)
        self.player_path_label.setText(str(audio_path))

    def clear_player_state(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self.current_audio_path = ""
        self.player_title_label.setText("未选择音频")
        self.player_path_label.setText("")
        self.player_time_label.setText("00:00 / 00:00")
        self.player_position_slider.blockSignals(True)
        self.player_position_slider.setRange(0, 0)
        self.player_position_slider.setValue(0)
        self.player_position_slider.blockSignals(False)
        self.player_position_slider.setEnabled(False)
        self.player_play_pause_btn.setEnabled(False)
        self.player_play_pause_btn.setText("播放")

    def load_audio_into_player(self, path: str, *, autoplay: bool) -> None:
        audio_path = Path(path)
        if not audio_path.exists():
            QMessageBox.warning(self, "播放", f"音频不存在: {path}")
            return
        self.update_player_metadata(str(audio_path.resolve()))
        self.player_position_slider.setEnabled(True)
        self.player_play_pause_btn.setEnabled(True)
        self.player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
        self.player_position_slider.setValue(0)
        self.update_player_time_label(position=0, duration=0)
        if autoplay:
            self.player.play()
        else:
            self.player.pause()

    def play_audio_in_player(self, path: str) -> None:
        self.load_audio_into_player(path, autoplay=True)

    def toggle_player_playback(self) -> None:
        if not self.current_audio_path:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop_player(self) -> None:
        self.player.stop()

    def _on_player_slider_pressed(self) -> None:
        self._player_user_dragging = True

    def _on_player_slider_released(self) -> None:
        self._player_user_dragging = False
        self.player.setPosition(self.player_position_slider.value())

    def _on_player_slider_moved(self, value: int) -> None:
        self.update_player_time_label(position=value)

    def _on_player_position_changed(self, position: int) -> None:
        if not self._player_user_dragging:
            self.player_position_slider.setValue(position)
        self.update_player_time_label(position=position)

    def _on_player_duration_changed(self, duration: int) -> None:
        self.player_position_slider.setRange(0, max(0, duration))
        self.player_position_slider.setEnabled(duration > 0)
        self.update_player_time_label(duration=duration)

    def _on_player_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player_play_pause_btn.setText("暂停")
        else:
            self.player_play_pause_btn.setText("播放")

    def _on_player_error(self, _error: QMediaPlayer.Error, error_string: str) -> None:
        if error_string:
            QMessageBox.warning(self, "播放", f"内置播放器无法播放该音频：{error_string}")

    def handle_input_table_selection_changed(self) -> None:
        self.handle_table_selection_changed(self.input_table, self.manifest_rows)

    def handle_table_selection_changed(self, table: QTableWidget, rows: list[dict[str, Any]]) -> None:
        row = self.selected_row(table, rows)
        if not row:
            return
        path = row.get("audio_path") or row.get("file_path") or row.get("音频路径")
        if not path:
            return
        self.load_audio_into_player(str(path), autoplay=False)

    def play_selected_audio(self, table: QTableWidget, rows: list[dict[str, Any]]) -> None:
        row = self.selected_row(table, rows)
        if not row:
            QMessageBox.warning(self, "播放", "请先选择音频。")
            return
        path = row.get("audio_path") or row.get("file_path") or row.get("音频路径")
        if not path:
            QMessageBox.warning(self, "播放", "没有可播放的音频路径。")
            return
        self.play_audio_in_player(str(path))

    def play_selected_asr_audio(self) -> None:
        self.play_selected_audio(self.asr_table, self.asr_rows)

    def play_selected_emotion_audio(self) -> None:
        self.play_selected_audio(self.emotion_table, self.emotion_rows)

    def play_last_audio(self) -> None:
        path = self.result_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "播放", "还没有可播放的合成音频。")
            return
        self.play_audio_in_player(path)

    def open_output_dir_for_role(self, role: str) -> None:
        role = role.strip()
        if not role:
            QMessageBox.warning(self, "输出目录", "请先选择角色。")
            return
        path = resolve_role_paths(role).output_dir / "tts_runs"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def open_role_output_dir(self) -> None:
        self.open_output_dir_for_role(self.role_combo.currentText())

    def open_output_overview_dir(self) -> None:
        self.open_output_dir_for_role(self.output_role_combo.currentText())

    def write_runtime_json(self, filename: str, payload: Any) -> Path:
        runtime_dir = PROJECT_ROOT / "voice_gui_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def append_log(self, text: str) -> None:
        target = self.active_log_view or self.log_view
        target.append(text)
        target.verticalScrollBar().setValue(target.verticalScrollBar().maximum())


def run_app() -> int:
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
