# 第四章正文与配图

[阅读版 HTML](../04-加速器架构.md) · [正文 Markdown](../04-加速器架构.md) · [写作大纲](../../archive/outlines/04-加速器架构.md)

八节、二十六小节，采用问题、机制、推导与设计取舍的教材体例。用同一 Q 投影贯穿计算与存储分析，再加入注意力、专家路径、封装和专用化；章内三道带解例题与八项实验。正文包含脚注和本地引用。

十四幅插图均提供 SVG、PNG 和 PDF。图号、图题和说明全部位于正文图片外；图内仅保留机制、子面板、单位与坐标标注。阅读 HTML 嵌入图像、公式与字体，可以离线阅读。

| 图 | 内容 | 文件 |
|---|---|---|
| 4-1 | 两种调用都使用同一份 32 MiB 权重。上方是一行输入，下方以部分条带示意 256 行；每行分摊的权重读取从 32 MiB 减至 128 KiB。 | [SVG](figure-4-1-reuse.svg) · [PNG](figure-4-1-reuse.png) · [PDF](figure-4-1-reuse.pdf) |
| 4-2 | 主机、显存与芯片内部的关系。实线表示数据经过缓存、局部缓冲、矩阵单元和累加存储，虚线表示主机提交工作。图按硬件功能分组。 | [SVG](figure-4-2-components.svg) · [PNG](figure-4-2-components.png) · [PDF](figure-4-2-components.pdf) |
| 4-3 | DaVinci 如何安排注意力计算：MTE/NDDMA、片上缓冲、Cube/AIC、Vector/AIV 与交接路径。 | [SVG](figure-4-davinci-attention-path.svg) · [PNG](figure-4-davinci-attention-path.png) · [PDF](figure-4-davinci-attention-path.pdf) |
| 4-4 | 三行三列的乘加阵列示意。输入沿行传递，权重沿列传递，每个乘加单元保留自己的部分和。这里用小阵列解释操作数复用。 | [SVG](figure-4-4-attention.svg) · [PNG](figure-4-4-attention.png) · [PDF](figure-4-4-attention.pdf) |
| 4-5 | 一个专家的 16 行计算块。每专家只有两行时，剩余十四行填零；每专家有 64 行时，可组成四个完整块，图中展示其中一块。 | [SVG](figure-4-5-precision.svg) · [PNG](figure-4-5-precision.png) · [PDF](figure-4-5-precision.pdf) |
| 4-6 | 相同 512 行有效输入在全部专家上的执行量。分散到 256 个专家后总计执行 4096 行，集中到八个专家时只执行 512 行。 | [SVG](figure-4-6-capacity.svg) · [PNG](figure-4-6-capacity.png) · [PDF](figure-4-6-capacity.pdf) |
| 4-7 | 同一注意力块在四种资源配置下的服务周期。矩阵、共享存储和指数三项分别比较，单独增加一种能力后，其他资源可能成为较长的一项。“×2”表示相应资源的吞吐能力提高一倍，横轴为完成同一计算块所需的时钟周期数。 | [SVG](figure-4-7-memory.svg) · [PNG](figure-4-7-memory.png) · [PDF](figure-4-7-memory.pdf) |
| 4-8 | 相同压缩权重的两条计算路径。先展开会形成 32 MiB 的 BF16 副本；低精度路径在计算过程中完成缩放和合并。压缩权重及缩放因子共 8.5 MiB。 | [SVG](figure-4-8-layout.svg) · [PNG](figure-4-8-layout.png) · [PDF](figure-4-8-layout.pdf) |
| 4-9 | 固定 24 GB 显存中的权重、工作区和 KV。8K 四请求与 16K 两请求可以容纳，8K 五请求超过虚线标出的容量上限。 | [SVG](figure-4-9-pipeline.svg) · [PNG](figure-4-9-pipeline.png) · [PDF](figure-4-9-pipeline.pdf) |
| 4-10 | 读取请求从发出到返回一直占用请求记录空间。多个独立请求交叠，才能在单次访问等待期间持续利用接口；图中只画四个代表请求。 | [SVG](figure-4-10-locality.svg) · [PNG](figure-4-10-locality.png) · [PDF](figure-4-10-locality.pdf) |
| 4-11 | 在每事务 128 bytes、返回延迟 500 ns 的题设下，增加在途请求数提高带宽上界，直到碰到接口本身的速率上限。 | [SVG](figure-4-11-interconnect.svg) · [PNG](figure-4-11-interconnect.png) · [PDF](figure-4-11-interconnect.pdf) |
| 4-12 | 每行前 256 bytes 是实际读取区间，相邻行起点相差 8192 bytes。128 行合计读取 32 KiB，行间灰色区域由步长跳过。 | [SVG](figure-4-12-specialization.svg) · [PNG](figure-4-12-specialization.png) · [PDF](figure-4-12-specialization.pdf) |
| 4-13 | 一个输入槽从发起到释放的完整生命周期。传输 64 tick，额外等待 128 tick，数据在 192 tick 就绪，再计算 128 tick，于 320 tick 释放。tick 是本算例的离散时间单位。 | [SVG](figure-4-13-roofline.svg) · [PNG](figure-4-13-roofline.png) · [PDF](figure-4-13-roofline.pdf) |
| 4-14 | 一个输入槽的四块时序。蓝条为传输，橙线为就绪，绿条为计算，浅灰为槽占用；上一块用完后才能再次发起，完成时刻为 1280 tick。tick 是本算例的离散时间单位。 | [SVG](figure-4-14-performance.svg) · [PNG](figure-4-14-performance.png) · [PDF](figure-4-14-performance.pdf) |

## 依据与复现

先阅读已有 calculations、survey、案例与实际实验记录，再编写正文。[reading-notes.md](reading-notes.md)说明采用范围；[sources.json](sources.json)锁定输入；[figure-data.json](figure-data.json)保存画图数据；[manifest.json](manifest.json)记录输出校验值。没有新跑 GPU 或模型质量实验。

```sh
python3 -m venv /tmp/ch04-book-venv
/tmp/ch04-book-venv/bin/pip install -r manuscripts/ch04/requirements.txt
/tmp/ch04-book-venv/bin/python manuscripts/ch04/build.py
/tmp/ch04-book-venv/bin/python manuscripts/ch04/check_browser.py
/tmp/ch04-book-venv/bin/python manuscripts/ch04/verify.py
```

构建需要 Python 3 和 Node.js。优先采用 macOS 中文字体，Linux 可安装 Noto Sans CJK 或传 `--font`。公式使用仓库已有 `../ch03/vendor/katex/` 的 KaTeX 0.16.11 和许可证，构建后字体和公式已嵌入 HTML。浏览器检查优先使用本机 Chrome，也支持 `CH04_CHROME` 指定可执行文件；没有浏览器时可安装 Playwright Chromium。

[validation.json](validation.json)包含大纲一致性、链接、图号、公式、来源与数值校验；[browser-validation.json](browser-validation.json)检查 1440 px 和 390 px 视口的图片、公式、目录与溢出。图布局另有自动边界检查与人工预览。

本次全章编辑的论证调整、有效数字规则与移入配套的细节见[编辑说明](editorial-revision.md)。

本轮全章重写增加节间递进、设计转折点和随文思考题；[教学推导与题解](teaching-notes.md)提供三槽流水、计算翻倍变体及容量、互联和 Roofline 边界。[证据记录](evidence-notes.md)集中保存版本、原始采集方法与数值检查。

[语言修订说明](language-revision.md)记录术语与中文句法调整；[段落衔接与配图修订](visual-revision.md)说明新增图的教学目的及对应推导。

## 当前阅读版配图（2026-09-10）

正文引用 38 幅图，其中图 4-4 至 4-7 为 4.1.3 节的芯片物理与能耗推导（`energy_physics.py`，输入为 `calculations/results/energy-ledger-book.json`）。4.6 介绍三家的定量架构演进，4.8 统一完成时间建模、同模型比较、实测校准和成本分析。[结构整合记录](../../research/ch04-section-integration-2026-09-10/README.md)记录内容分工；[量化计算](../../calculations/research/architecture-evolution-quantitative/README.md)保存复算依据。

| 图号 | 内容 | SVG | PNG | PDF |
| --- | --- | --- | --- | --- |
| 4-1 | 两种调用都使用同一份 32 MiB 权重。上方是一行输入，下方以部分条带示意 256 行；每行分摊的权重读取从 32 MiB 减至 128 KiB。 | [SVG](figure-4-1-reuse.svg) | [PNG](figure-4-1-reuse.png) | [PDF](figure-4-1-reuse.pdf) |
| 4-2 | 主机、显存与芯片内部的关系。实线表示数据经过缓存、局部缓冲、矩阵单元和累加存储，虚线表示主机提交工作。图按硬件功能分组。 | [SVG](figure-4-2-components.svg) | [PNG](figure-4-2-components.png) | [PDF](figure-4-2-components.pdf) |
| 4-3 | DaVinci 如何安排注意力计算：MTE/NDDMA、片上缓冲、Cube/AIC、Vector/AIV 与交接路径。 | [SVG](figure-4-davinci-attention-path.svg) | [PNG](figure-4-davinci-attention-path.png) | [PDF](figure-4-davinci-attention-path.pdf) |
| 4-4 | 上：同一驱动门经数百微米横向连线传送信号，要给沿线分布电容充电；折叠后改为数微米的垂直连接。下：电压从 0.85 V 降到 0.55 V 使动态功耗降到 0.42；功耗降到 0.75 而投影面积降到 0.60，功率密度反而升到 1.25。 | [SVG](figure-4-energy-wire.svg) | [PNG](figure-4-energy-wire.png) | [PDF](figure-4-energy-wire.pdf) |
| 4-5 | 上：各级存储与链路每 byte 能耗，对数坐标。下：Qwen3-8B 单请求 8K decode 一步的能量分账（权重 0.481 J、KV 0.038 J、计算 0.015 J），以及同样 16.345 GB 全部来自某一层次时的能量。 | [SVG](figure-4-energy-ladder.svg) | [PNG](figure-4-energy-ladder.png) | [PDF](figure-4-energy-ladder.pdf) |
| 4-6 | 封装俯视示意：两颗达到光罩上限的 die 居中，八堆 HBM 沿两侧边缘排列，中介层承载 die 与 HBM 之间以及 die 之间的连线。每堆带宽由 1,024 根引脚与引脚速率决定，每堆容量由层数与每层容量决定。 | [SVG](figure-4-energy-package.svg) | [PNG](figure-4-energy-package.png) | [PDF](figure-4-energy-package.pdf) |
| 4-7 | 持续频率与峰值频率之比随每 FLOP 能耗 e 与预算 b 之比的变化：电压不变时为 b/e，电压随频率下降时为 (b/e)^(1/3)。e/b 不超过 1 时峰值频率可以持续；e/b 为 1.5 时两条曲线分别给出 0.67 与 0.87。 | [SVG](figure-4-energy-power-cap.svg) | [PNG](figure-4-energy-power-cap.png) | [PDF](figure-4-energy-power-cap.pdf) |
| 4-8 | 三行三列的乘加阵列示意。输入沿行传递，权重沿列传递，每个乘加单元保留自己的部分和。这里用小阵列解释操作数复用。 | [SVG](figure-4-matrix-array.svg) | [PNG](figure-4-matrix-array.png) | [PDF](figure-4-matrix-array.pdf) |
| 4-9 | 一个专家的 16 行计算块。每专家只有两行时，剩余十四行填零；每专家有 64 行时，可组成四个完整块，图中展示其中一块。 | [SVG](figure-4-3-expert-rows.svg) | [PNG](figure-4-3-expert-rows.png) | [PDF](figure-4-3-expert-rows.pdf) |
| 4-10 | 相同 512 行有效输入在全部专家上的执行量。分散到 256 个专家后总计执行 4096 行，集中到八个专家时只执行 512 行。 | [SVG](figure-4-expert-padding-total.svg) | [PNG](figure-4-expert-padding-total.png) | [PDF](figure-4-expert-padding-total.pdf) |
| 4-11 | 同一注意力块在四种资源配置下的服务周期。矩阵、共享存储和指数三项分别比较，单独增加一种能力后，其他资源可能成为较长的一项。“×2”表示相应资源的吞吐能力提高一倍，横轴为完成同一计算块所需的时钟周期数。 | [SVG](figure-4-4-attention.svg) | [PNG](figure-4-4-attention.png) | [PDF](figure-4-4-attention.pdf) |
| 4-12 | 相同压缩权重的两条计算路径。先展开会形成 32 MiB 的 BF16 副本；低精度路径在计算过程中完成缩放和合并。压缩权重及缩放因子共 8.5 MiB。 | [SVG](figure-4-5-precision.svg) | [PNG](figure-4-5-precision.png) | [PDF](figure-4-5-precision.pdf) |
| 4-13 | RTX 4090 的 24 GB 显存中的权重、工作区和 KV。8K 四请求与 16K 两请求可以容纳，8K 五请求超过虚线标出的容量上限。 | [SVG](figure-4-6-capacity.svg) | [PNG](figure-4-6-capacity.png) | [PDF](figure-4-6-capacity.pdf) |
| 4-14 | 读取请求从发出到返回一直占用请求记录空间。多个独立请求交叠，才能在单次访问等待期间持续利用接口；图中只画四个代表请求。 | [SVG](figure-4-memory-inflight.svg) | [PNG](figure-4-memory-inflight.png) | [PDF](figure-4-memory-inflight.pdf) |
| 4-15 | 每事务 128 bytes、返回延迟 500 ns 时，增加在途请求数会提高带宽上界，直到碰到 RTX 4090 或 RTX 5090 显存接口本身的速率上限。 | [SVG](figure-4-7-memory.svg) | [PNG](figure-4-7-memory.png) | [PDF](figure-4-7-memory.pdf) |
| 4-16 | 每行前 256 bytes 是实际读取区间，相邻行起点相差 8192 bytes。128 行合计读取 32 KiB，行间灰色区域由步长跳过。 | [SVG](figure-4-8-layout.svg) | [PNG](figure-4-8-layout.png) | [PDF](figure-4-8-layout.pdf) |
| 4-17 | 一个输入槽从发起到释放的完整生命周期。传输 64 tick，额外等待 128 tick，数据在 192 tick 就绪，再计算 128 tick，于 320 tick 释放。一个 tick 为 B200 SM 的一个时钟周期。 | [SVG](figure-4-slot-lifetime.svg) | [PNG](figure-4-slot-lifetime.png) | [PDF](figure-4-slot-lifetime.pdf) |
| 4-18 | 一个输入槽的四块时序。蓝条为传输，橙线为就绪，绿条为计算，浅灰为槽占用；上一块用完后才能再次发起，完成时刻为 1280 tick。一个 tick 为 B200 SM 的一个时钟周期。 | [SVG](figure-4-9-pipeline.svg) | [PNG](figure-4-9-pipeline.png) | [PDF](figure-4-9-pipeline.pdf) |
| 4-19 | 两个输入槽使用相同时间尺度。前两块可提前发起，但第三块到 512 tick 才就绪，第二块在 448 tick 已结束，留下 64 tick 空闲。横轴一个 tick 为 B200 SM 的一个时钟周期；各行对应一个数据块，灰色表示输入槽占用，蓝色表示传输，绿色表示计算，竖标记表示数据就绪。 | [SVG](figure-4-pipeline-two.svg) | [PNG](figure-4-pipeline-two.png) | [PDF](figure-4-pipeline-two.pdf) |
| 4-20 | 三个输入槽提前发起前三块，第一槽释放后接收第四块。矩阵单元从 192 连续计算到 704 tick，第四槽不再缩短完成时间。横轴一个 tick 为 B200 SM 的一个时钟周期；各行对应一个数据块，灰色表示输入槽占用，蓝色表示传输，绿色表示计算，竖标记表示数据就绪。 | [SVG](figure-4-pipeline-three.svg) | [PNG](figure-4-pipeline-three.png) | [PDF](figure-4-pipeline-three.pdf) |
| 4-21 | 矩阵与向量单元通过完整行组交接。QK 产生分数，Softmax 产生概率，PV 使用概率后释放槽；另一槽容纳相邻行组，使不同组可以重叠推进。 | [SVG](figure-4-matrix-vector-handoff.svg) | [PNG](figure-4-matrix-vector-handoff.png) | [PDF](figure-4-matrix-vector-handoff.pdf) |
| 4-22 | 计算都放在 die 0 时，要跨 die 读取 die 1 上的 32 GiB 权重。B200 的跨 die 读取受 die 1 的 HBM 限制，约 8.6 ms，与本地读取相同；昇腾 910C 受 die 间链路限制，约 127 ms。 | [SVG](figure-4-10-locality.svg) | [PNG](figure-4-10-locality.png) | [PDF](figure-4-10-locality.pdf) |
| 4-23 | QK、Softmax、PV 的两槽执行时间线：两个缓冲槽让不同查询行组交错推进。 | [SVG](figure-4-davinci-attention-timeline.svg) | [PNG](figure-4-davinci-attention-timeline.png) | [PDF](figure-4-davinci-attention-timeline.pdf) |
| 4-24 | 一次 8 KiB 传输的启动与传输时间。固定启动开销 2 μs，NVLink 从 A100 的每方向 300 GB/s 换成 H100 的 450 GB/s，只缩短很薄的蓝色传输项。 | [SVG](figure-4-11-interconnect.svg) | [PNG](figure-4-11-interconnect.png) | [PDF](figure-4-11-interconnect.pdf) |
| 4-25 | 一次 2 MiB 传输在相同启动条件下的时间。蓝色传输项占主要部分，换成 H100 的 NVLink 带来更显著的收益；本图纵轴范围与上一图分别标注。 | [SVG](figure-4-large-message.svg) | [PNG](figure-4-large-message.png) | [PDF](figure-4-large-message.pdf) |
| 4-26 | 同一 V100 上的 Q 投影。4,096 行时，Tensor Core 大幅缩短矩阵计算；一行时，两条计算路径都短于权重传输。上下两组使用分别标注的时间单位。 | [SVG](figure-4-evolution-tensor-budget.svg) | [PNG](figure-4-evolution-tensor-budget.png) | [PDF](figure-4-evolution-tensor-budget.pdf) |
| 4-27 | 同一个 4096×4096 权重的存储与读取预算。每 32 个低精度值共享一个一字节缩放因子。位宽从 BF16 降至 MXFP4 后，权重与缩放因子合计从 32 MiB 减至 8.5 MiB。 | [SVG](figure-4-evolution-precision.svg) | [PNG](figure-4-evolution-precision.png) | [PDF](figure-4-evolution-precision.pdf) |
| 4-28 | 从异步拷贝、TMA 到 TMEM，专门部件承担更多数据准备与累加状态管理。蓝色表示数据存储与搬运，橙色表示矩阵计算，紫色表示累加结果。 | [SVG](figure-4-evolution-nvidia-path.svg) | [PNG](figure-4-evolution-nvidia-path.png) | [PDF](figure-4-evolution-nvidia-path.pdf) |
| 4-29 | 同一个 3×3 卷积的两条准备路径。显式展开形成九倍大小的中间矩阵；MTE 在片上组织窗口，省去 HBM 上 6.891 MiB 的中间结果写读。 | [SVG](figure-4-evolution-img2col.svg) | [PNG](figure-4-evolution-img2col.png) | [PDF](figure-4-evolution-img2col.pdf) |
| 4-30 | 早期同核分工、910B／910C 的独立控制，以及 950 增加的 CV 直接通路。计算单元的配比与结果交换的路径共同决定融合算子的效率。 | [SVG](figure-4-evolution-ascend.svg) | [PNG](figure-4-evolution-ascend.png) | [PDF](figure-4-evolution-ascend.pdf) |
| 4-31 | 同一注意力块的两次交接。外层交换接口承担写出和读入共 256 KiB；直接 CV 通路传递两个 64 KiB 张量，将这笔流量移出外层接口。在 1.024 μs 内完成交接，对应直接通路的带宽需求为 128 GB/s。 | [SVG](figure-4-evolution-cv-budget.svg) | [PNG](figure-4-evolution-cv-budget.png) | [PDF](figure-4-evolution-cv-budget.pdf) |
| 4-32 | 统一内存负责整机的数据容量，Dynamic Caching 管理 GPU 局部资源，M5 的 Neural Accelerator 增加 GPU 内专用计算能力。三者分别进入容量、驻留和计算时间的分析。 | [SVG](figure-4-evolution-apple.svg) | [PNG](figure-4-evolution-apple.png) | [PDF](figure-4-evolution-apple.pdf) |
| 4-33 | 固定权重的同时，各请求仍独立读取 KV。题设每请求保留 8K 上下文，batch 从 13 起 KV 读取超过共享权重读取。 | [SVG](figure-4-12-specialization.svg) | [PNG](figure-4-12-specialization.png) | [PDF](figure-4-12-specialization.pdf) |
| 4-34 | 独立只读权重改变两条存储路径。上方权重与 KV 争用 HBM；下方 ROM 提供权重，HBM 存放可写状态。箭头表示读取，KV 还需写入新状态。 | [SVG](figure-4-rom-paths.svg) | [PNG](figure-4-rom-paths.png) | [PDF](figure-4-rom-paths.pdf) |
| 4-35 | 同一 Q 投影的计算与访存耗时随输入行数的变化。计算量按行数增长，片外访问同时包含固定权重和增长的输入输出；从 179 行起计算项较长。 | [SVG](figure-4-13-roofline.svg) | [PNG](figure-4-13-roofline.png) | [PDF](figure-4-13-roofline.pdf) |
| 4-36 | 同一 Qwen3-8B 的两种阶段预算。单请求 decode 更直接反映读取带宽，4K prefill 的矩阵预算更直接反映匹配精度的矩阵速率。两图横轴分别标注所计算的时间。 | [SVG](figure-4-evolution-convergence.svg) | [PNG](figure-4-evolution-convergence.png) | [PDF](figure-4-evolution-convergence.pdf) |
| 4-37 | RTX PRO 6000 的投影总耗时。每个条件测十一轮、每轮十六次调用，取每轮平均耗时的中位数；计时包含提交与同步。“复用”表示多次调用读取相同权重地址；“轮换”表示调用之间更换权重地址。 | [SVG](figure-4-14-performance.svg) | [PNG](figure-4-14-performance.png) | [PDF](figure-4-14-performance.pdf) |
| 4-38 | 相同四个条件下另行采集的 DRAM 读取计数。单行均约 32 MiB，256 行复用为 256 bytes、轮换约 32.1 MiB。访问计数与常规计时分别测量。“复用”与“轮换”分别表示保持和更换权重地址。 | [SVG](figure-4-performance-traffic.svg) | [PNG](figure-4-performance-traffic.png) | [PDF](figure-4-performance-traffic.pdf) |

[2026-09-11 全章复核](../../research/ch04-whole-chapter-review-2026-09-11/README.md)：合并跨节重复，重排固定权重推导，补齐概念定义与交接算例参数。
