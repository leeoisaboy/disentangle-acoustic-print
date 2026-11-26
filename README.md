<p align="center">
  <a href="https://github.com/leeoisaboy/disentangle-acoustic-print">
    <img src="docs/assets/title_banner.zh.png" alt="Disentangle-Acoustic-Print 跨音色时序解耦与四轨白盒声学工作台" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-Acoustic_Print_Studio-0ea5e9?style=flat-square"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.4.0-6366f1?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-30%20passing-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Disentangle-Acoustic-Print (v0.4)</h1>

<p align="center"><strong>零训练、纯 NumPy 的跨音色时序解耦与流式声学指纹 SDK，配备实时四轨白盒声学工作台。</strong></p>

> **白盒语音检索三部曲（White-Box Speech Retrieval Trilogy）**
> 1. **[Disentangle-Acoustic-Print](https://github.com/leeoisaboy/disentangle-acoustic-print)**（本仓库）：零训练、纯 NumPy 几何解耦、层探针与超低延迟端侧粗召回 / 白盒四轨示波器（CPU RTF `0.013–0.045`）。
> 2. **[AudioLLM-Latent-KWS](https://github.com/leeoisaboy/audiollm-latent-kws)**：录 1–3 条样音即可跨方言高精度定位的帧级 DTW / AudioLLM 潜空间检词系统（连续流严格 FA/hr 门禁）。
> 3. **[Dialect-CrossModal-Align](https://github.com/leeoisaboy/dialect-crossmodal-align)**：不录样音、只写音素/IPA 拼写即可零样本检索方言新词的跨模态对齐框架。

---

## v0.4 架构总览

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="Disentangle-Acoustic-Print v0.4 四阶段架构总览图" width="1100" />
</p>

v0.4 将原有的合成轨迹原型全面推进到**三套公开真实语料 × 三轨真实特征前端**的严格跨说话人实证验收，并收口为通用的两层架构：
- **纯 NumPy 核心 SDK（`pip install disentangle-acoustic-print`）**：零模型训练、零硬性深度学习依赖，通过帧率无关的 `FeatureFrontend` 协议接入任意前端（端侧 Log-Mel / ONNX / WavLM / Whisper），4 步完成 `前端抽帧 → 低秩软正交去音色 → 模板注册与背景 FA/hr 校准 → 多尺度滑窗定位`。
- **四轨白盒工作台（`pip install "disentangle-acoustic-print[studio]"`）**：60 FPS 浏览器本地示波器，将波形速度着色、动力学 CUSUM 变点、音色剥离能量、等弧长标尺与四分区指纹条形码对齐在同一条时间轴上实时透视。

---

## 正在运行的界面（Acoustic Print Studio）

<p align="center">
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/">
    <img src="docs/assets/studio-demo.gif" alt="Acoustic Print Studio 实时四轨白盒声学示波器动图" width="1100" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/disentangle-acoustic-print/">▶ 进入在线 Acoustic Print Studio</a></strong></p>

| 轨道 | 页面中的内容 | 对应模块 |
| --- | --- | --- |
| Track 01 | 速度着色波形、柔和匹配光带、Temporal IoU NMS 抑制 | `OnlineMatcher`、`decompose().speed` |
| Track 02 | 瞬时速度、双峰、零均值 CUSUM 变点、轨迹折返角（白盒诊断层） | `extract_evidence`、`compute_cusum_curve` |
| Track 03 | 线性刻度 vs 等弧长刻度对照、说话人子空间剥离能量表 | `arc_length_resample`、`SpeakerSubspaceProjector` |
| Track 04 | Static、Delta、Speed、Accel 四分区连续指纹纹理 | `fingerprint`、`decompose_fingerprint_barcode` |

---

## 真实公开数据五组实证研究（MagicHub 沪语 · KeSpeech 多方言 · LibriPhrase 英语）

所有实证均在严格**说话人零重叠（0 speaker overlap）**协议下完成（`SpeakerSubspaceProjector` 仅拟合背景与注册说话人，严禁接触任何测试说话人），同一批音频横向对比三类前端：
- **端侧轻量轨**：80 维纯 NumPy Log-Mel（100 Hz，流式安全无分段 CMVN 漂移）
- **通用自监督轨**：`microsoft/wavlm-large`（50 Hz，1024 维，逐层探针锁定 L14 + 深层对照 L21）
- **跨语种大模型轨**：`openai/whisper-tiny` Encoder（50 Hz，384 维，逐层探针锁定 L4）

完整原始指标保存在 [reports/real_data_benchmark_v04.json](reports/real_data_benchmark_v04.json)，门禁自动校验结果见 [reports/release_gate_verdict_v04.json](reports/release_gate_verdict_v04.json)。

### 1. 层探针与说话人正交投影帕累托前沿（Study 1 & Study 2C）

<p align="center">
  <img src="docs/assets/layer_probe_and_projector_pareto.png" alt="Study 1 层探针与 Study 2C 正交投影秩扫描帕累托前沿" width="1080" />
</p>

- **Study 1（层探针锁定最佳解耦层）**：`layerwise_probe` 在 WavLM-Large 全部 24 层上自动测出中层内容/说话人方差比峰值位于 **L14（0.956）**，深层对照 **L21（0.817）** 明显回落；实测跨说话人检索 L14 在全部三套语料上均大幅领先 L21（如 L3 英语 Top-1 `88.1%` vs `61.9%`）。Whisper-tiny 4 层编码器单调上升并在 **L4（1.927）** 达峰，零训练跨说话人孤立词 Top-1 分别达 **L3 英语 96.0%（EER 1.6%）**、**L2 多方言 61.1%（EER 13.9%）**、**L1 沪语 56.1%（EER 20.0%）**。
- **Study 2A（四视图单挑）**：在原始 Log-Mel 与 WavLM 上，去均值后的 **`residual`（$H_t - \mu_H$）是跨说话人最抗音色漂移的单视图**，比静态均值 `static` 高出 **+10 至 +25 个百分点**（如 L1 WavLM `32.2%` vs `16.7%`，L2 Mel `35.0%` vs `9.4%`）；而纯 1D 速度剖面 `velocity` 单独区分力极弱（`4%–9%`）。因此 v0.4 新增 `fingerprint(..., preset="content")`（`0.45 Static + 0.35 Residual + 0.20 Delta`）。
- **Study 2C（正交音色投影低秩律）**：在 **3/3 套公开语料** 上，低秩 `SpeakerSubspaceProjector`（`rank=2..4` 或 `variance_threshold + max_rank + shrinkage`）显著降低跨说话人 EER；但高秩（`rank=16, 32`）会过度剥离发音内容导致掉点。

| 公开语料（说话人零重叠） | 纯均值基线 EER (`WavLM L14`) | `r=0` 原始指纹 EER | 低秩投影最佳 EER (`r=2..5`) | EER 绝对下降 | `Whisper L4` 零训练 Top-1 / EER |
| --- | ---: | ---: | ---: | ---: | ---: |
| **L1 沪语 (MagicHub)** | 41.7% | 37.2% | **32.2%** (`r=2`) | **-9.5 pp** | **56.1% / 20.0%** |
| **L2 多方言 (KeSpeech)** | 27.2% | 25.0% | **20.7%** (`r=2`, Top-1 +11.1 pp) | **-6.5 pp** | **61.1% / 13.9%** |
| **L3 英语 (LibriPhrase)** | 12.5% / 18.2% | 18.2% | **15.5%** (`r=2`, Top-1 +5.7 pp) | **-2.7 pp** | **96.0% / 1.6%** |

### 2. 结构化证据在线负结果审计 & 等弧长防噪增益（Study 3 & Study 2B）

<p align="center">
  <img src="docs/assets/evidence_audit_and_tempo_guard.png" alt="Study 3 结构化证据在线权重审计与 Study 2B 等弧长防噪护栏增益" width="1080" />
</p>

- **Study 3（诚实报告负结果：`evidence_weight` 在线单调降分）**：在全部 **3 语种 × 3 前端 × 干净/带噪** 组合上，将 `evidence_weight` 从 `0.00` 增至 `0.10 / 0.25 / 0.50` 均会导致在线跨说话人 Top-1 与 EER **单调恶化**（例如 L3 Whisper Clean 从 `90.9%` 降至 `87.5% → 67.0% → 30.7%`）。v0.4 据此将全栈默认 `evidence_weight` 严格收口为 **`0.0`**，而将 `extract_evidence()`（零均值 CUSUM、双峰、折返角）明确定位为**白盒诊断与 Studio 四轨可视化层**。
- **Study 2B（等弧长重采样的“虚假弧长”隐患与 VAD/低通修复）**：在 `0.6×` 慢速、`1.5×` 快速不变调变速以及首尾填充 150 ms 噪声静音（10 dB SNR）的真实场景下，未防护的 `arc_raw` 会被噪声抖动制造的虚假弦长拉偏；开启 v0.4 内置的轻量端点裁剪与双向一阶低通护栏（`arc_length_resample(..., trim_silence=True, smooth_alpha=0.5)`）后，跨说话人 Top-1 在变速与带噪条件下一致回升 **+6.1 至 +12.8 个百分点**（如 L1 WavLM `0.6×` 从 `13.3% → 24.4%`，L2 Whisper 带噪从 `6.1% → 18.9%`）。

### 3. 连续未切分语音流校准与流式等价性（Study 4 & Study 5）

- **Study 4（连续流校准跃升与诚实能力边界）**：
  - 未校准的孤立模板 + 全局 40 词 `argmax` 滑窗在连续语音流上会出现严重的枢纽词偏置与上下文漂移（Event AUC 仅 `0.017–0.028`）。
  - 引入 v0.4 的**同上下文模板 + 逐词中位窗长 + `EnrollmentBank.calibrate_background()` 逐词背景 Z-Norm** 后，连续流 Event AUC @ IoU≥0.5 在 L1 沪语跃升至 **0.7509**（+73.4 pp），在 L2 多方言跃升至 **0.8103**（+78.2 pp），在 L3 英语达 **0.6267**。
  - **诚实边界说明**：零训练单向量余弦在 40 词连续流上能高效完成粗召回与事件排序（AUC `0.75–0.81`），但在极端严格的 `FA/hr ≤ 1–2` 下 Recall@IoU≥0.5 仅 `7.2%–12.2%`，**未达 80% 严格门禁**；若业务需要未切分长流上的高召回低误报，建议将本库作为前端粗筛/诊断层，与姐妹仓库 [AudioLLM-Latent-KWS](https://github.com/leeoisaboy/audiollm-latent-kws) 的帧级 DTW 精排级联使用。
- **Study 5（流式环形缓冲等价性与单核 CPU 延迟）**：关闭分段均值漂移（`normalize_utterance=False`）后，100 ms 分块流式推进与离线全流扫描在 L1/L2 真实长流上达成**数学完全等价（Pearson `r = 1.0000`，最大绝对误差 `0.0`，峰值时间戳误差 `0` 帧）**；单核 CPU 实时率 **RTF = `0.0135`（Mel80） / `0.0437`（WavLM-Large 1024 维）**，远低于 `0.2` 门禁。

### 4. 发布门禁自动验收结果（`python benchmarks/verify_release_gates.py`）

| 门禁 | 验收标准 | 实测结果 | 状态 |
| --- | --- | --- | :---: |
| **Gate 1 跨说话人泛化** | ≥ 2 套公开语料上投影+等弧长显著降 EER | L1 `-9.5 pp`、L2 `-6.6 pp`（**2/3 优于纯均值**；L3 较 `r=0` 指纹亦降 `-2.7 pp`） | **PASS** |
| **Gate 2 连续流可用性** | FA/hr ≤ 2 时 Event Recall@IoU≥0.5 ≥ 80% | 校准后 Event AUC 跃升至 `0.751–0.810`，但零训练单向量 Recall@0.5 为 `7.2%–8.9%`（诚实记录能力边界） | **BOUNDARY** |
| **Gate 3 流式等价与延迟** | 离线/流式相关系数 ≥ 0.99，单核 CPU RTF < 0.2 | `r = 1.0000`，`peak_err = 0`，`RTF = 0.0135–0.0450` | **PASS** |
| **Gate 4 零侵入通用性** | 切换语种/前端零改核心代码，全套单测 100% 通过 | 3 语种 × 3 前端零改核心代码，**30/30 单元测试全过** | **PASS** |

---

## 快速上手（4 步纯 NumPy 核心 SDK）

### 安装

```bash
# 核心 SDK（仅依赖 NumPy，适合端侧 / 服务端轻量部署）
pip install -e .

# 可选：启动四轨白盒工作台（FastAPI + HTML5 Canvas）
pip install -e ".[studio]"

# 可选：接入 ONNX 或 PyTorch/Transformers 预训练前端
pip install -e ".[onnx]"
pip install -e ".[torch]"
```

### 4 步极简调用

```python
from disentangle_acoustic_print import (
    EnrollmentBank,
    MelDeltaExtractor,
    OnlineMatcher,
    SpeakerSubspaceProjector,
)

# 1. 统一帧率无关前端（也可替换为 HiddenStateBridge / OnnxEncoderFrontend / PyTorchSSLFrontend）
frontend = MelDeltaExtractor(n_mels=80, hop_ms=10.0, include_delta=False, normalize_utterance=False)
enroll_frames, fps = frontend.extract_with_rate(enroll_wav, sample_rate=16000)

# 2. 低秩软正交去音色（严禁使用测试说话人拟合）
projector = SpeakerSubspaceProjector.fit(
    bg_trajectories, bg_speaker_ids, variance_threshold=0.80, max_rank=4, shrinkage=0.75
)

# 3. 模板注册 + 无标注背景流 FA/hr 动态阈值校准（默认 evidence_weight=0.0）
bank = EnrollmentBank(projector=projector, preset="content", trim_silence=True, smooth_alpha=0.5)
bank.enroll("open-light", [enroll_frames])
bank.calibrate_background(unlabeled_bg_streams, target_fa_per_hour=1.0, frame_rate_hz=fps)

# 4. 毫秒/自适应窗长多尺度流式匹配 + Temporal IoU NMS
matcher = OnlineMatcher(
    bank, window_length="auto", hop_ms=40.0, frame_rate_hz=fps,
    scales=(0.75, 1.0, 1.25), iou_threshold=0.4, use_calibration=True,
)
matches = matcher.match(stream_frames, top_k=3)
```

### 启动本地 Studio 与测试

```bash
# 运行 30 项全量单元测试
pytest -v

# 复现发布门禁核验
python benchmarks/verify_release_gates.py

# 启动四轨白盒工作台（FastAPI WebSocket 模式）
acoustic-print-studio --mode fastapi --port 8000
```

更多契约与模块说明请参阅 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) 与 [docs/MODULE_MAP.md](docs/MODULE_MAP.md)。
