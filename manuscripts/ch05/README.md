# 第五章正文与插图

[5.2–5.3 试改阅读版](../05-算子与运行时.md) · [整章阅读版 HTML](../05-算子与运行时.md) · [正文 Markdown](../05-算子与运行时.md) · [写作大纲](../../archive/outlines/05-算子与运行时.md)

正文按设备执行、单算子、算子链、编译器、运行时和完整请求六节展开，含 23 小节、12 道带解例题和 9 项章末实验。5.2 节把局部存储布局与切分维度分成两个小节；5.1 和 5.5 新增复制路径、流与事件、主机流水、图重放和形状分桶的机制图。每组例题从条件推导结果，再改变条件形成设计判断。段落通过数据依赖、资源限制与前一方案尚未解决的问题衔接；机制图与时间线先说明工作如何执行，再与公式和数值对应。AKG 的多面体编译通过 Halide／TVM 风格的循环变换、依赖与存储寿命解释。

当前正文共 40 幅插图，均提供 SVG 与 PNG；5.2–5.3 的十三幅试改图另提供 PDF，按 420 pt（约 148 mm）宽设计，主体标签 12 pt、辅助标签不低于 11 pt。图号和完整图题位于正文 caption。阅读版内嵌图片、KaTeX 公式和字体，可离线阅读。

| 图 | 内容 | 文件 |
| --- | --- | --- |
| 5-1 | 提交、执行与结果可用的时间 | [SVG](figure-5-1-execution.svg) · [PNG](figure-5-1-execution.png) |
| 5-2 | 相邻输出怎样复用同一行输入 | [SVG](figure-5-reuse-steps.svg) · [PNG](figure-5-reuse-steps.png) · [PDF](figure-5-reuse-steps.pdf) |
| 5-3 | 输入块更换而输出累加器保留 | [SVG](figure-5-tile-working-set.svg) · [PNG](figure-5-tile-working-set.png) · [PDF](figure-5-tile-working-set.pdf) |
| 5-4 | 局部容量与矩阵重读 | [SVG](figure-5-2-tiles.svg) · [PNG](figure-5-2-tiles.png) · [PDF](figure-5-2-tiles.pdf) |
| 5-5 | 同一局部存储预算容纳的工作集 | [SVG](figure-5-tile-residency.svg) · [PNG](figure-5-tile-residency.png) · [PDF](figure-5-tile-residency.pdf) |
| 5-6 | 补齐行跨度如何分散 bank 请求 | [SVG](figure-5-3-banks.svg) · [PNG](figure-5-3-banks.png) · [PDF](figure-5-3-banks.pdf) |
| 5-7 | RMSNorm 拆分后的归约与输入重读 | [SVG](figure-5-4-reduction.svg) · [PNG](figure-5-4-reduction.png) · [PDF](figure-5-4-reduction.pdf) |
| 5-8 | 融合前后中间结果经过的路径 | [SVG](figure-5-fusion-path.svg) · [PNG](figure-5-fusion-path.png) · [PDF](figure-5-fusion-path.pdf) |
| 5-9 | 融合边界与中间读写 | [SVG](figure-5-5-boundaries.svg) · [PNG](figure-5-5-boundaries.png) · [PDF](figure-5-5-boundaries.pdf) |
| 5-10 | 两个缓冲槽的交替使用 | [SVG](figure-5-buffer-slots.svg) · [PNG](figure-5-buffer-slots.png) · [PDF](figure-5-buffer-slots.pdf) |
| 5-11 | 同样四块数据的串行与双缓冲执行 | [SVG](figure-5-6-fusion-buffer.svg) · [PNG](figure-5-6-fusion-buffer.png) · [PDF](figure-5-6-fusion-buffer.pdf) |
| 5-12 | 完整中间矩阵与逐块统计量 | [SVG](figure-5-attention-storage.svg) · [PNG](figure-5-attention-storage.png) · [PDF](figure-5-attention-storage.pdf) |
| 5-13 | 在线 Softmax 如何合并两个块 | [SVG](figure-5-7-online-softmax.svg) · [PNG](figure-5-7-online-softmax.png) · [PDF](figure-5-7-online-softmax.pdf) |
| 5-14 | 注意力块大小在访问量与更新次数之间的取舍 | [SVG](figure-5-8-attention-tradeoff.svg) · [PNG](figure-5-8-attention-tradeoff.png) · [PDF](figure-5-8-attention-tradeoff.pdf) |
| 5-15 | 循环层级决定临时数据的保存时间 | [SVG](figure-5-9-polyhedral.svg) · [PNG](figure-5-9-polyhedral.png) |
| 5-16 | 保存量化结果如何减少后续读取 | [SVG](figure-5-10-quantization.svg) · [PNG](figure-5-10-quantization.png) |
| 5-17 | 调用比例与平均执行时间 | [SVG](figure-5-11-feedback.svg) · [PNG](figure-5-11-feedback.png) |
| 5-18 | 主机提交与设备执行的对应关系 | [SVG](figure-5-12-runtime.svg) · [PNG](figure-5-12-runtime.png) |
| 5-19 | 图重放的固定收益与输入复制时间 | [SVG](figure-5-13-graph-copy.svg) · [PNG](figure-5-13-graph-copy.png) |
| 5-20 | 准备成本与累计执行时间 | [SVG](figure-5-14-specialization.svg) · [PNG](figure-5-14-specialization.png) |
| 5-21 | 从等待全部投影到逐块开始激活 | [SVG](figure-5-15-persistent.svg) · [PNG](figure-5-15-persistent.png) |
| 5-22 | 局部加速引起关键路径切换 | [SVG](figure-5-16-critical-path.svg) · [PNG](figure-5-16-critical-path.png) |
| 5-23 | 完整请求的逐轮配对时间差 | [SVG](figure-5-17-request.svg) · [PNG](figure-5-17-request.png) |




[取材记录](reading-notes.md)说明已读 calculations、survey、论文与实验的采用范围。[sources.json](sources.json)锁定输入，[figure-data.json](figure-data.json)保存图数据，[figure-index.json](figure-index.json)将当前图号映射到稳定文件名。新增图片后图号已顺延；旧文件名与数据键保留，避免破坏复算引用。[manifest.json](manifest.json)记录输出。未运行新 GPU 基准，未将已有 partial 实验改为完成。

## 重建

从仓库根目录执行，需要 Python 3、Node.js 和中文字体：

```sh
python3 -m venv /tmp/ch05-book-venv
/tmp/ch05-book-venv/bin/pip install -r manuscripts/ch05/requirements.txt
/tmp/ch05-book-venv/bin/python manuscripts/ch05/build.py
/tmp/ch05-book-venv/bin/python manuscripts/ch05/verify.py
```

生成器优先使用 macOS Arial Unicode，Linux 可用 Noto CJK，或指定 `--font /path/to/font`。来源变化时构建停止，需审阅后更新锁。KaTeX 0.16.11 与许可证随 vendor 保存。

## 检查

[validation.json](validation.json)包含目录、练习、引用、输入输出校验及主要算例复核；[math-validation.json](math-validation.json)记录公式渲染；[figure-layout-check.json](figure-layout-check.json)记录文字边界。图像已目视检查。[browser-validation.json](browser-validation.json)记录桌面 1440 px 和手机 390 px 的图片加载、公式与横向溢出检查。

可选浏览器检查：安装 `playwright` 后运行 `python manuscripts/ch05/check_reading.py`。macOS 优先使用独立无头 Chrome；其他平台需先安装 Playwright Chromium。此项不读取个人浏览器会话。

## 5.2–5.3 图文试改

本轮新增六幅过程图，重画七幅既有图，并按“对象与操作 → 机制 → 数量关系 → 限制条件”重排正文。`teaching_revision.py` 集中绘制这十三幅图；`build.py` 继续读取既有算例与实验数据，并生成整章和独立试读版。手机端保留图中文字大小，可在图内左右滑动。

[修改说明与修改前快照](../../archive/reviews/ch05-52-53-teaching-2026-09-10/README.md) · [试读 PDF](../../archive/reviews/ch05-52-53-teaching-2026-09-10/sections-5.2-5.3.pdf)。

## 当前阅读版配图（2026-09-12）

以下图号以当前正文顺序为准；前面的旧版图表记录仅用于历史对照。

| 图号 | 内容 | SVG | PNG | PDF |
|---|---|---|---|---|
| 5-1 | H2D 把输入从主机的锁页缓冲搬进显存，D2H 把结果搬回主机；权重加载一次后留在显存，kernel 直接读取显存中的输 | [SVG](figure-5-copy-paths.svg) | [PNG](figure-5-copy-paths.png) | [PDF](figure-5-copy-paths.pdf) |
| 5-2 | 复制流依次传入各批输入，计算流读取它们。计算批 0 要等“批 0 已传完”事件；批 2 要重新写入槽 A，必须等“批 0 | [SVG](figure-5-stream-event.svg) | [PNG](figure-5-stream-event.png) | [PDF](figure-5-stream-event.pdf) |
| 5-3 | CPU 提交、H2D 输入复制、kernel 执行与 D2H 结果返回依次发生。上方 kernel 执行 20 μs，结 | [SVG](figure-5-1-execution.svg) | [PNG](figure-5-1-execution.png) | [PDF](figure-5-1-execution.pdf) |
| 5-4 | 三条分别表示 SM 的寄存器、共享内存与线程槽总量，蓝、绿为两个驻留线程块的占用。寄存器恰好填满，共享内存剩余 34 K | [SVG](figure-5-sm-residency.svg) | [PNG](figure-5-sm-residency.png) | [PDF](figure-5-sm-residency.pdf) |
| 5-5 | 上行是生产者 warp 发出的异步复制，下行是消费者 warp 的计算；蓝、绿分别为槽 A、B。实线箭头为“满”栅栏，复 | [SVG](figure-5-warp-pipeline.svg) | [PNG](figure-5-warp-pipeline.png) | [PDF](figure-5-warp-pipeline.pdf) |
| 5-6 | 先算一个输出，再算相邻输出。蓝色表示所用的 A 行，橙色表示所用的 W 列，绿色表示本次得到的输出元素。图中矩阵缩小为示 | [SVG](figure-5-reuse-steps.svg) | [PNG](figure-5-reuse-steps.png) | [PDF](figure-5-reuse-steps.pdf) |
| 5-7 | 固定当前 m×n 输出块，沿 K 依次搬入对应的 A、W 块，反复更新同一份部分和，全部累加结束后才写回。A、W 输入为 | [SVG](figure-5-tile-working-set.svg) | [PNG](figure-5-tile-working-set.png) | [PDF](figure-5-tile-working-set.pdf) |
| 5-8 | 每个点标出输出块形状。扩大输出块能减少跨接口的重复读取，但需要更多局部存储。虚线为三份矩阵各经过一次的 128 MiB； | [SVG](figure-5-2-tiles.svg) | [PNG](figure-5-2-tiles.png) | [PDF](figure-5-2-tiles.pdf) |
| 5-9 | 条形总长均表示 RTX PRO 6000 一个 SM 的 100 KB 共享内存。24 KiB 的工作集连同每块 1 K | [SVG](figure-5-tile-residency.svg) | [PNG](figure-5-tile-residency.png) | [PDF](figure-5-tile-residency.pdf) |
| 5-10 | 同一列的 32 个不同字由 32 个 lane 同时请求。上半图行跨度为 32 个字，请求集中到同一 bank；下半图补 | [SVG](figure-5-3-banks.svg) | [PNG](figure-5-3-banks.png) | [PDF](figure-5-3-banks.pdf) |
| 5-11 | 上行由一组处理完整一行，输入保留到归一化结束；下行把一行分为八段，先求局部和，再合并，最后重读输入并归一化。每行 409 | [SVG](figure-5-4-reduction.svg) | [PNG](figure-5-4-reduction.png) | [PDF](figure-5-4-reduction.pdf) |
| 5-12 | 切输出行 $M$ 或输出列 $N$，各执行者得到不同位置的完整结果，需要时再拼起来；切归约维 $K$，各执行者得到同一输 | [SVG](figure-5-split-axes.svg) | [PNG](figure-5-split-axes.png) | [PDF](figure-5-split-axes.pdf) |
| 5-13 | 同一行 4096 个 BF16 元素的平方和与精确值的距离，单位为 FP32 的 ULP。上半图比较六种切分档位；下半图 | [SVG](figure-5-reduction-order.svg) | [PNG](figure-5-reduction-order.png) | [PDF](figure-5-reduction-order.pdf) |
| 5-14 | 上半图的完整 T 经过一次写出与一次读回；下半图中局部片段 t 直接传给乘法。两种方式仍读取 G、U 并写出 Z，图中省 | [SVG](figure-5-fusion-path.svg) | [PNG](figure-5-fusion-path.png) | [PDF](figure-5-fusion-path.pdf) |
| 5-15 | 每少保存一个 24 MiB 中间量，就少一次写出和一次读入，共 48 MiB。G、U 为 BF16，最终输出占 1 by | [SVG](figure-5-5-boundaries.svg) | [PNG](figure-5-5-boundaries.png) | [PDF](figure-5-5-boundaries.pdf) |
| 5-16 | 块 0 在 1.29–1.57 μs 使用槽 A，此后槽 A 已经空出；块 2 要等搬移器在 2.58 μs 搬完块 1 | [SVG](figure-5-buffer-slots.svg) | [PNG](figure-5-buffer-slots.png) | [PDF](figure-5-buffer-slots.pdf) |
| 5-17 | 两个输入槽交替复用，让搬移与计算重叠。H100 一个 SM 上每块搬移 1.29 μs、计算 0.28 μs，资源独立， | [SVG](figure-5-6-fusion-buffer.svg) | [PNG](figure-5-6-fusion-buffer.png) | [PDF](figure-5-6-fusion-buffer.pdf) |
| 5-18 | 上半图保存完整 S、P，两份 FP32 矩阵各占 256 MiB，写出与读回合计 1 GiB。下半图只传递已处理部分的最 | [SVG](figure-5-attention-storage.svg) | [PNG](figure-5-attention-storage.png) | [PDF](figure-5-attention-storage.pdf) |
| 5-19 | 两个块的分数分别为 0、ln 2，值分别为 1、3。最大值增大后，将旧指数和与旧加权值同时乘以 1/2，再加上新块的贡献 | [SVG](figure-5-7-online-softmax.svg) | [PNG](figure-5-7-online-softmax.png) | [PDF](figure-5-7-online-softmax.pdf) |
| 5-20 | 快速缓冲为 RTX PRO 6000 一个线程块的 99 KB 共享内存，序列长 8192、头维度 128，无掩码；每个 | [SVG](figure-5-8-attention-tradeoff.svg) | [PNG](figure-5-8-attention-tradeoff.png) | [PDF](figure-5-8-attention-tradeoff.pdf) |
| 5-21 | 循环层次决定临时数据需要保存多久。外层选输出块，创建 16 KiB 累加器；内层 ko 反复读取 A、W 块，全部归约结 | [SVG](figure-5-9-polyhedral.svg) | [PNG](figure-5-9-polyhedral.png) | [PDF](figure-5-9-polyhedral.pdf) |
| 5-22 | 同样两个部分和，先相加得到零，再做 SiLU 仍为零；先对各部分做 SiLU 再相加，得到约 0.4621。两个数说明把 | [SVG](figure-5-activation-order.svg) | [PNG](figure-5-activation-order.png) | [PDF](figure-5-activation-order.pdf) |
| 5-23 | 一行分成两个块，后一块中的 10 决定整行 scale。第一项要先按这一 scale 映射，再舍入到格式允许的值，最后反 | [SVG](figure-5-quantization-scale.svg) | [PNG](figure-5-quantization-scale.png) | [PDF](figure-5-quantization-scale.pdf) |
| 5-24 | 两种方案都先读完整输入以确定每行的 scale。保存 FP8 结果后 12 个列块合计重读 192 MiB；融合方案重读 | [SVG](figure-5-10-quantization.svg) | [PNG](figure-5-10-quantization.png) | [PDF](figure-5-10-quantization.pdf) |
| 5-25 | 形状 A 占比超过 2/3 时，新实现的总执行时间更短。A、B 原耗时均为 10 μs，新实现分别为 5、20 μs；调 | [SVG](figure-5-11-feedback.svg) | [PNG](figure-5-11-feedback.png) | [PDF](figure-5-11-feedback.pdf) |
| 5-26 | 每段主机准备 20 μs。串行执行时两种资源轮流工作；流水后主机准备下一段时加速器计算当前段，每 20 μs 完成一段； | [SVG](figure-5-host-pipeline.svg) | [PNG](figure-5-host-pipeline.png) | [PDF](figure-5-host-pipeline.pdf) |
| 5-27 | 普通提交时，主机为每个 kernel 发起一次 launch；图重放时，主机只发起一次 graph launch，加速器 | [SVG](figure-5-launch-vs-graph.svg) | [PNG](figure-5-launch-vs-graph.png) | [PDF](figure-5-launch-vs-graph.pdf) |
| 5-28 | 普通提交的 3 次 FFN 实测。上行为主机 kernel launch API，下行为加速器 kernel；横轴从本段 | [SVG](figure-5-12-runtime.svg) | [PNG](figure-5-12-runtime.png) | [PDF](figure-5-12-runtime.pdf) |
| 5-29 | 3 次 FFN 融合激活链后，主机 kernel launch 15 次，加速器相应执行 15 个 kernel。数据来 | [SVG](figure-5-runtime-1.svg) | [PNG](figure-5-runtime-1.png) | [PDF](figure-5-runtime-1.pdf) |
| 5-30 | 3 次 graph launch 对应 18 个加速器 kernel。图重放减少主机提交次数，加速器仍执行原图各节点；本 | [SVG](figure-5-runtime-2.svg) | [PNG](figure-5-runtime-2.png) | [PDF](figure-5-runtime-2.pdf) |
| 5-31 | 先融合再重放，主机发起 3 次图执行，加速器执行 15 个 kernel。融合减少加速器 kernel 数，图重放减少主 | [SVG](figure-5-runtime-3.svg) | [PNG](figure-5-runtime-3.png) | [PDF](figure-5-runtime-3.pdf) |
| 5-32 | 图执行时读取记录的地址 G。新输入位于 X 时先复制到 G；上游直接写 G 时沿用同一缓冲，省去中间复制 | [SVG](figure-5-graph-address.svg) | [PNG](figure-5-graph-address.png) | [PDF](figure-5-graph-address.pdf) |
| 5-33 | 橙色为准备，蓝色为额外输入复制，绿色为加速器计算。普通方式 40 μs；图的 2 MiB 输入约 27 μs，16 Mi | [SVG](figure-5-13-graph-copy.svg) | [PNG](figure-5-13-graph-copy.png) | [PDF](figure-5-13-graph-copy.pdf) |
| 5-34 | 256 行的调用补齐到 512 行的桶，1536 行的调用补齐到 2048 行的桶，2048 行恰好落在桶的边界上。灰色 | [SVG](figure-5-shape-buckets.svg) | [PNG](figure-5-shape-buckets.png) | [PDF](figure-5-shape-buckets.pdf) |
| 5-35 | 同一组调用反复执行时，最省时的策略随复用次数变化。曲线采用例 5-12 的形状、处理率和准备时间，不同策略适用的整数调用 | [SVG](figure-5-14-specialization.svg) | [PNG](figure-5-14-specialization.png) | [PDF](figure-5-14-specialization.pdf) |
| 5-36 | 粗粒度执行先完成八块投影，再启动八块激活。按 RTX PRO 6000 的矩阵峰值与显存带宽，每块投影约 12.8 μs | [SVG](figure-5-15-persistent.svg) | [PNG](figure-5-15-persistent.png) | [PDF](figure-5-15-persistent.pdf) |
| 5-37 | 矩阵与向量资源独立、缓冲充足。每个任务另计 0.7 μs 调度与通知，首块投影完成后即可激活；投影首尾相接，激活只在每块 | [SVG](figure-5-persistent-blocks.svg) | [PNG](figure-5-persistent-blocks.png) | [PDF](figure-5-persistent-blocks.pdf) |
| 5-38 | 收尾等待 A、B 两条分支。A 从 60 μs 缩短至 15 μs 后，较慢分支由 A 切换为 B，请求从 80 μs  | [SVG](figure-5-16-critical-path.svg) | [PNG](figure-5-16-critical-path.png) | [PDF](figure-5-16-critical-path.pdf) |
| 5-39 | 全局驻留与逐层逻辑读取分开比较。上方仅计全局 KV 容量，下方读取包含全局条目、局部窗口和索引；两栏各自采用相同横轴尺度 | [SVG](figure-5-v41-traffic.svg) | [PNG](figure-5-v41-traffic.png) | [PDF](figure-5-v41-traffic.pdf) |
| 5-40 | 同轮替换前的请求时间减去替换后的请求时间，正值表示替换后更快。RTX PRO 6000 上的 Qwen3-8B，7239 | [SVG](figure-5-17-request.svg) | [PNG](figure-5-17-request.png) | [PDF](figure-5-17-request.pdf) |
