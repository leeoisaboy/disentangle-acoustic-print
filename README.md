<p align="center">
  <a href="https://github.com/leeoisaboy/disentangle-acoustic-print">
    <img src="docs/assets/title_banner.zh.png" alt="Disentangle-Acoustic-Print 跨音色时序解耦与四轨白盒声学工作台" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-Acoustic_Print_Studio-0ea5e9?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-23%20passing-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Disentangle-Acoustic-Print</h1>

<p align="center"><strong>把一段冻结语音轨迹拆成可以解释的时序视图，并让每个算子实时显示出来。</strong></p>

<img align="right" width="190" src="docs/assets/logo.png" alt="Disentangle-Acoustic-Print logo" />

## 正在运行的界面

<p align="center">
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/">
    <img src="docs/assets/studio-demo.gif" alt="Acoustic Print Studio 实时四轨白盒声学示波器动图" width="1100" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/disentangle-acoustic-print/">进入在线 Acoustic Print Studio</a></strong></p>

这段动图来自真实页面，不是概念稿。在线版在浏览器内运行，支持麦克风流、拖入 WAV、语速与音色干扰滑杆、帧置乱对照、双语音速注册和遥测 JSON 导出。静态页面不需要后端，音频在本地处理。

四轨把同一份遥测数据拆成四种观察方式。

| 轨道 | 页面中的内容 | 对应实现 |
| --- | --- | --- |
| Track 01 | 速度着色波形、多尺度捕获框、temporal IoU 抑制 | `OnlineMatcher`、`decompose().speed` |
| Track 02 | 瞬时速度、双峰、零均值 CUSUM 变点、轨迹折返角 | `extract_evidence`、`compute_cusum_curve` |
| Track 03 | 线性刻度与等弧长刻度对照、音色子空间剥离能量 | `arc_length_resample`、`SpeakerSubspaceProjector` |
| Track 04 | Static、Delta、Speed、Accel 四分区指纹条码 | `fingerprint`、`decompose_fingerprint_barcode` |

<details>
<summary>展开四轨全页截图</summary>
<p align="center">
  <img src="docs/assets/studio-full.png" alt="Acoustic Print Studio 四轨全页界面" width="1100" />
</p>
</details>

## 为什么做这个项目

智能家居和适老化设备会不断遇到新的方言指令。同一个词换一个人说，音色变了，语速也变了。检索系统如果只保存一个平均向量，容易把说话人的音色当成词的一部分；如果只保留帧间运动，发音人的节奏差异又会把同一句话推得很远。

这个项目研究冻结语音表征中的内容与音色如何在时间轴上混在一起。仓库把一段隐状态轨迹拆成静态、运动、残差、速度四类视图，用结构化证据补齐边界与节奏，用多尺度在线匹配定位事件。核心检索与几何算子只依赖 NumPy，不训练额外匹配器。

Studio 把这些中间量直接画出。波形、变点、峰、重采样刻度、音色剥离能量和四分区指纹都能在同一时间轴上核对，算法出错时也能看见错在哪个环节。

## 项目追问什么

1. 静态均值与一阶运动，哪一种视图更能跨说话人保留内容。
2. 同一个人改变语速后，怎样用弧长参数化把节奏从形状里拿走。
3. 音色差异表现为特征子空间旋转时，低秩正交投影能否降低泄漏。
4. 结构化证据与声学指纹怎样融合，什么时候有效，什么时候只增加噪声。
5. 在线扫描怎样避免同一个事件附近反复返回高度重叠的窗口。
6. 这些判断能否被同一份遥测包还原，并在浏览器里逐帧检查。

## 实验证据

当前代码提供一套可复算的研究基线。确定性合成夹具负责验证接口、数值契约和失败模式，真实语音结论仍需要外部音频与冻结特征。此前本地探索日志留下的证据如下。

| 探针 | 结果 | 说明 |
| --- | ---: | --- |
| 真实时序形状 Top-1 | 21.32% | 局部运动携带可用的词条信息 |
| 打乱帧序后的 Top-1 | 14.60% | 破坏顺序后下降 6.72 点 |
| 单向量检索上限 | 58.05% | 静态与速度视图的参考上限 |
| 结构化证据 oracle 上限 | 70.43% 至 71.76% | 证据摘要存在明显余量 |

这些数字来自原始本地探索日志。它们说明帧序确实包含信息，也说明结构化证据有潜力。oracle 上限不能直接当作在线增益，真实收益仍取决于校准、候选池和融合权重。

当前版本能够直接验证以下工程契约。

- 四类时序视图都有独立实现，弧长重采样可以作为默认时序归一化。
- 零均值 CUSUM、局部峰筛选和最小间距抑制替代了退化的统计量。
- 指纹与模板统一 L2 归一化，增量注册按累计样本数更新。
- 证据校准只在注册数据变化时重算，在线混合检索直接复用缓存。
- 多尺度窗口经过 temporal IoU NMS，返回结果不会再被相邻窗口挤满。
- 音频桥、流式滚动缓冲区和四轨遥测均有端到端测试。

## 方法

<p align="center">
  <img src="docs/assets/pipeline.zh.png" alt="从隐状态轨迹到四轨在线示波器的流程" />
</p>

给定轨迹 $H \in \mathbb{R}^{T \times D}$，仓库提供四种互补视图。

| 视图 | 构造 | 作用 |
| --- | --- | --- |
| 静态中心 | $\mu_H = T^{-1}\sum_t H_t$ | 跨音色的内容锚点 |
| 差分形状 | $\Delta H_t = H_{t+1}-H_t$，可先做等弧长重采样 | 发音过渡方向 |
| 残差轨迹 | $H_t-\mu_H$ | 相对词条中心的运动位置 |
| 速度剖面 | $\lVert \Delta H_t \rVert_2$ | 快慢、长短与局部能量证据 |

结构化证据包含响应强度、零均值 CUSUM 变点、两个分隔峰、速度统计、转角统计和归一化边界轮廓。证据向量按注册样本做 Z-score 校准，再与指纹余弦得分融合。

工程上有五个关键选择。

- `arc_length_resample()` 按累积弦长等距采样，降低说话速度变化对时序描述的影响。
- `SpeakerSubspaceProjector` 从说话人均值估计低秩音色子空间，再把该子空间从轨迹中投影掉。
- `EnrollmentBank` 保存单位指纹、累计均值和证据样本。校准器缓存在下一次注册前一直有效。
- `OnlineMatcher` 支持多个窗口长度，并用 temporal IoU NMS 抑制重叠结果。
- `RollingAcousticStream` 把 16 kHz 音频、滚动环形缓冲区、实时注册和四轨遥测串成同一条数据链。

## 使用 Studio

在线版直接打开浏览器即可。页面进入后会自动演示多尺度捕获，也可以点击方言词条、调整语速与音色干扰，或开启麦克风流。

本地静态运行不需要安装后端依赖。

```bash
python -m http.server 8765 --directory studio
```

浏览器打开 `http://127.0.0.1:8765`。

需要 WebSocket 服务时启动 FastAPI 入口。

```bash
python -m pip install -e ".[studio]"
python -m studio.app --mode fastapi --port 8000
```

Hugging Face Spaces 或 Gradio 流式页面使用另一个入口。

```bash
python -m pip install -e ".[spaces]"
python -m studio.app --mode gradio --port 7860
```

## 快速开始

```bash
python -m venv .venv
python -m pip install -e .
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。

```python
from disentangle_acoustic_print import (
    EnrollmentBank,
    OnlineMatcher,
    SpeakerSubspaceProjector,
    make_cross_voice_dataset,
    make_stream,
)

dataset = make_cross_voice_dataset(dimensions=16)
projector = SpeakerSubspaceProjector.fit(
    dataset["sequences"],
    dataset["speaker_ids"],
    rank=2,
)

frames = dataset["sequences"][0]
bank = EnrollmentBank(projector=projector, evidence_weight=0.2)
bank.enroll("phrase-a", [frames])
print(bank.search(frames)[0])

stream = make_stream(
    dimensions=16,
    target=frames.mean(axis=0),
    insert_start=10,
    target_length=len(frames),
)
locator = OnlineMatcher(
    bank,
    window_length=len(frames),
    scales=(0.75, 1.0, 1.25),
    hop=2,
    iou_threshold=0.4,
)
print(locator.match(stream)[0])
```

命令行提供确定性演示与四轨遥测。

```bash
disentangle-acoustic-print demo
disentangle-acoustic-print evidence
disentangle-acoustic-print fingerprint
disentangle-acoustic-print online
disentangle-acoustic-print telemetry
```

## 这个原型有什么意义

它可以作为自监督语音表征的诊断工具。换层、换模型或换池化方式时，四类视图能把内容保持、音色泄漏和时序稳定性拆开报告。`layerwise_probe` 还能直接比较不同特征层的内容方差与说话人方差。

它也可以作为低资源词条检索的基线。新词只需要少量注册轨迹。声学定位给出候选，结构化证据补充边界与节奏线索，在线匹配器在连续流里找事件。Studio 把这套基线变成可操作的实验台，滑杆改变语速、音色和融合权重，画面立即显示各轨响应。

它最有价值的地方在于保留失败。合成数据可以复现，数值契约可以测试，阴性结果可以复核。一个能解释失败的声音检索基线，比一个只在演示里正确的结果更接近可用的研究工具。

## 实验边界

- 仓库只接收 `[time, dimension]` 隐状态轨迹，不包含模型权重、私有清单或远程服务凭证。
- 合成夹具与 Studio 预设用于接口检查和交互演示，不代表真实吴语或普通话语料上的识别率。
- `SpeakerSubspaceProjector` 只能使用允许参与建模的注册说话人拟合，查询说话人不应进入估计。
- temporal IoU 阈值与窗口尺度仍需要按实际帧率、词长和事件密度调整。
- 历史日志中的 oracle 上限不是线上 Top-1，不能直接用于产品承诺。
- 浏览器麦克风需要安全上下文或 localhost 权限，首次开启由浏览器弹窗确认。

## 仓库结构

```text
disentangle_acoustic_print/
  geometry.py       时序分解、弧长重采样、音色子空间投影
  probing.py        层探针、排序指标与分组校准
  evidence.py       CUSUM 变点、峰提取、结构化证据校准
  fingerprint.py    单位指纹、混合检索、多尺度在线定位
  synthetic.py      带语速形变与正交旋转的确定性夹具
  audio_bridge.py   16 kHz WAV 编解码、Mel 轨迹提取与模型桥
  telemetry.py      四轨白盒遥测序列化
  stream_engine.py  环形缓冲区、双语音速注册与实时匹配
  cli.py            JSON 演示与命令行入口
studio/
  index.html        无后端四轨 Canvas 示波器
  app.py            FastAPI WebSocket 与统一启动入口
  gradio_app.py     Hugging Face Spaces 流式界面
docs/
  DATA_CONTRACT.md
  MODULE_MAP.md
  assets/           横幅、LOGO、流程插图、界面截图与动态预览
scripts/
  make_assets.py    生成 README 视觉资产
tests/
  23 个数值、音频、遥测与接口回归测试
```

## 复现与测试

```bash
python -m compileall -q .
python -m pytest -q
```

当前测试结果为 `23 passed`。输入契约见 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)，模块职责见 [docs/MODULE_MAP.md](docs/MODULE_MAP.md)。

## 许可

代码使用 MIT License。音频、模型权重和派生特征保留各自原始许可。

**一个值得留下的声音检索基线，应当同时解释它为什么命中，以及它为什么失败。**
