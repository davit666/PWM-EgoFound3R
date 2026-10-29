# EgoFound3R 训练重建记录

论文：[arXiv 2610.01210v1](https://arxiv.org/pdf/2610.01210v1)。
上游：[MINT-SJTU/EgoFound3R](https://github.com/MINT-SJTU/EgoFound3R)，审计版本
`264f4df9fd42536aae4dba982d18272cc3334077`。

**当前是工程重建，不是已完成的论文数值复现。** 上游公开推理架构和权重，未公开训练
入口、原始数据转换器、精确 split manifest 或评测入口。这里将论文可确定部分与重建
假设分开。现有视频证明预训练推理可用，不能证明训练和 Table 1 等结果已经复现。

## 论文要求与实现位置

| 项目 | 论文要求 | 当前实现/边界 |
|---|---|---|
| 网络 | 冻结 VGGT-Ω，学习 hand prompts/adapters/heads | 复用发布的 builders；训练启动检查 backbone requires_grad |
| 时间轴 | RGB at H，scene at G⊆H，stride 1–5 | 原生 multirate mapping；scene 标签与 G 使用同一 indices |
| 输出 | 21 joints + 195 MANO points；相机/深度/尺度 | 直接从 high_rate 取训练输出，不用可视化导出的张量 |
| 几何 Eq.3 | 分别归一化预测/GT 的点尺度，exp(s) L1 − αs | objectives.geometry_loss；尺度估计采用 valid 点平均范数 |
| 可见性 Eq.4 | masked BCE | objectives；缺标签保持 mask=false |
| 接触 Eq.5 | exp(c) BCE − αc | objectives；物体+另一只手合并，无 partner id |
| 距离 Eq.6 | SmoothL1(log(1+1000d)) | 输入 d 单位米，原生 joints/vertices 各自监督 |
| 标签几何 | point-to-triangle；joint 18mm / vertex 14mm | geometry.py；不是最近顶点距离；未知 partner 不生成负样本 |
| 2D Eq.7 | weighted heatmap BCE、offset、reprojection | labels.py + objectives；像素/格点坐标与上游 decoder 一致 |
| Scene Eq.8 | confidence depth、K、归一化 translation、rotation geodesic | scene.py；尺度来自反投影点，轨迹统一第一相机 gauge |
| 训练 Table 5 | AdamW、3200 steps、warmup100、cosine、bf16、clip1 | train_prepared.py 单设备入口，支持梯度累积/恢复 |
| 分布式 | 8 H20、8/device、ZeRO-2、effective64 | 已在8 H100上验证microbatch1累积8、effective64的2步训练/恢复；未做3200步正式训练 |
| 手几何评测 | Raw/RR/PA/Sim3/W/WA | evaluation.py 函数；未跑正式 benchmark |
| 属性评测 | per-window P/R/F1；distance All/Pred/GT | evaluate_prepared.py；macro average 排除 undefined |
| 时序/场景评测 | 速度/加速度、ATE/rotation、AbsRel/RMSE/δ1 | evaluation.py 函数；待正式数据集接入 |
| 推理后处理 | H→R、195→778、scale alignment、stitching | 保留上游实现；训练 checkpoint 采用原 checkpoint contract |

## 必须公开记录的重建假设

`configs/training_reconstruction_example.json` 是**示例实验配置**，不是作者 recipe。
α=0.1、所有 task 权重1、heatmap positive weight10、stride均匀、seed0、训练 clip
采样、Gaussian heatmap sigma1 均为重建选择。几何平均范数、scene confidence 的具体
robust L1 形式、场景尺度的有效点集合也需要与作者核对。论文未披露它们的精确实现。
论文提到 temporal motion supervision，但未给出独立 temporal loss 的公式/权重；当前
只实现时间评测，未冒充已经恢复原 temporal objective。

不能把 image in-frame 当成几何 visibility，也不能把缺物体网格标成“不接触”。手套
到 MANO 的标定可以给出明确拓扑对应，但力/压力不是这里的几何 distance 标签。
力预测属于后续模型扩展，本轮不改变原论文输出头。

## Prepared clip contract v1

JSONL 每行必须包含 `dataset, source_episode_id, split, path`，推荐 `sha256`。
同一 source episode 不得跨 split，防止相邻窗口泄漏；精确论文 split 尚不可恢复。
NPZ 禁止 pickle；包括：

- `rgb`: uint8 `[T,H,W,3]`，已同步、去畸变，训练应为256×256。
- `timestamp`: 秒 `[T]`，严格递增；`frame_present`: bool `[T]`，v1只接受完整窗口。
- `target__NAME`, `valid__NAME`: 每项显式 mask。几何 offset 的 mask 去掉 xyz 维，
  其他 mask 与 value 同形。无标签不添加该项；无效值不会参与 loss。
- `joint_offset`: `[T,2,21,3]`，`vertex_offset`: `[T,2,195,3]`，相机坐标、米、减腕根。
- `joint/vertex_{visibility,contact,distance}`: `[T,2,N]`，二值或距离米。
- `heatmap`: `[T,2,21,Hh,Wh]`；`offset2d`: `[T,2,21,2,Hh,Wh]`。
- `reprojection`: 像素 `[T,2,21,2]`，配 `projection_K`: `[T,3,3]`。
- `scene_depth`: `[T,H,W]` 米，`scene_K`: `[T,3,3]`，`scene_pose`: `[T,4,4]` W2C。
  保存 H 时间轴，训练抽到什么 stride 就同步 gather 到 G，禁止预先固定 stride 标签。

每个原始 adapter 仍需核对：左右手排序、21 joints 顺序、MANO778→固定195、单位、
W2C/C2W、RGB时间戳、图像去畸变及缩放后的 K、三角面 winding/坐标、物体姿态、
visibility 定义、depth 有效掩码。原始 RGB 不经标定处理，不能直接视为可训练数据。

## 训练和验证入口

```bash
python scripts/pwm/train_prepared.py --config configs/pwm_baseline.toml \
  --recipe configs/training_reconstruction_example.json --index /path/index.jsonl \
  --output /path/run --preflight-only
python scripts/pwm/train_prepared.py --config configs/pwm_baseline.toml \
  --recipe configs/training_reconstruction_example.json --index /path/index.jsonl \
  --output /path/run --initialize /path/official_marker.pt
python scripts/pwm/train_prepared.py --config configs/pwm_baseline.toml \
  --recipe configs/training_reconstruction_example.json --index /path/index.jsonl \
  --output /path/run --resume /path/run/latest.pt
python scripts/pwm/evaluate_prepared.py --config configs/pwm_baseline.toml \
  --recipe configs/training_reconstruction_example.json --index /path/index.jsonl \
  --checkpoint /path/run/latest.pt --output /path/run/eval.json
```

`--initialize` 是 posttraining，省略它则随机初始化可训练部分（仍加载冻结 backbone）。
不能把短程 posttraining 称为完整从头复现。Checkpoint 包含非 backbone 权重、immutable
architecture contract、optimizer、step、Python/NumPy/Torch/CUDA RNG、recipe、config、index hash。
未带内容 hash 的 index 无法防止 NPZ 被外部替换。`--stop-after` 可用于恢复测试且不改 LR
schedule。单卡保存采用临时文件原子替换；分布式入口已通过同world-size的短程分片恢复。

## 数据与评测验收门槛

原始训练集：H2O、HOT3D Aria、OakInk-v2、TACO、HOI4D、ARCTIC、STERA-10M。
前六集 eval windows 数分别283、400、400、400、461、434，共2378；STERA仅训练。
训练 sequences 数分别138、136、565、839、1431、226，STERA all。作者未提供具体 id。
重新划分只能注明为我们自己的复现划分，不能直接与论文表格声称相同测试集。

2026-10-29 在 tac2 发现旧表六个 `/share/project/...` 路径不可见。实测找到以下位置：

| 数据集 | 当前 tac2 可读位置 | 实际存储形态 |
|---|---|---|
| H2O | `/share-mixed/project/eai_pwm/datasets/H2O` | tar.gz；MANO标签抽样可读 |
| HOT3D Aria | `/share-mixed/project/eai_pwm/datasets/Hot3DAria` | sequence目录 |
| OakInk-v2 | `/share-mixed/project/eai_pwm/datasets/OakInk-v2` | tar + anno_preview，版本须核对 |
| TACO | `/share-mixed/project/eai_pwm/datasets/TACO` | 手/相机/物体/视频各自ZIP；成员索引可读 |
| ARCTIC | `/share-mixed/project/eai_pwm/datasets/ARCTIC` | downloads/data + body_models |
| HOI4D | 尚未在本次候选路径找到 | 不代表其他挂载或服务器没有 |
| STERA-10M | `/share-mixed/projset/eai_pwm/datasets/Stera-10M` | session RGB/HDF5/calibrations |

STERA 本地实际文件采用 `hand-pose/frame_idx, side, joints_cam_lidar,...` 稀疏 detection
表；当前[官方 schema](https://www.fpvlabs.ai/stera/docs/process/concepts/hdf5-schema)
描述的是 per-frame left/right 数组，不能直接照搬。scene adapter 明确按本地 schema
读取 RGB 时间戳，再以20ms上限匹配相机和深度，应用 `R_optical_to_link`，将 C2W
转为 W2C。RGB/depth 都按标定重采样到同一256×256射线网格。首轮 smoke 不使用
尚未核实左右手/拓扑含义的稀疏手标注。

目录存在、文件数/大小验收、标签 schema 可用、转换完成、训练验证通过是不同状态。
原始标注审计见 `docs/results/training_source_audit_20261029/`，这是有界抽样而非全量验收。

完整完成条件：7集 adapter 实测、固定可追溯 split、所有可用监督接通、实际完整模型
backward/恢复/推理回载、8卡ZeRO-2训练3200步、正式2378窗口评测及与论文差异分析。
在这些条件满足前，只报告具体通过的检查和实际训练步数。

## 已执行验证（2026-10-29）

- 本地26项测试通过（22项训练重建、4项既有相机/可视化回归），覆盖ZeRO-2有效batch、
  portable export、scalar gate广播恢复、几何/距离/掩码梯度和评测定义。
- tac2 H100 GPU0，真实 STERA session 的31–33帧，256×256，官方权重初始化。
  原生场景输出完成两步 AdamW；第1步保存后重启进程，从 optimizer/RNG checkpoint
  恢复到第2步。训练 loss 的梯度范数分别57.63、61.85（裁剪前），均有限。
- 新进程通过原 `load_marker_model_weights` 加载636个状态项，stride5前向完成，
  终态 exit_code=0。记录在 [training_smoke_20261029](results/training_smoke_20261029/)。
- eval使用同一个训练窗口，仅验证权重回载及输出有限性；不能解释为泛化指标。训练为
  stride1，eval为stride5，二者的 loss 不可直接比较。当前三帧窗口的stride5只有一个
  G anchor，rotation loss为0属预期，不能视为相机精度。
- 该试验覆盖scene/K/translation/rotation/metric，未覆盖真实hand/contact loss；
  TACO adapter 的验证单独记录，不能由STERA结果推断。
- TACO `(dust, roller, pan)/20230927_032` 第30–32帧也完成两步真实训练及断点恢复。
  使用[数据集原始 MANO 实现](https://github.com/leolyliu/TACO-Instructions)
  `fd3b5a6b39565ca6935a91d35d5e91dde03c5142`，flat_hand_mean=True、center_idx=0；
  world坐标经发布的 W2C 变换，工具/目标模型厘米转米。已目视核对
  [GT投影](results/training_smoke_taco_20261029/label_projection.png)。
- TACO 已验证 joint/vertex geometry、contact、distance、heatmap、offset、projection
  的非零梯度，记录包含各模块 gradient norm。裁剪前总梯度范数约2429.82和4562.24，
  按论文规则裁剪到1；这只是短程连通性检查，不说明该重建权重配比已经合适。
  未生成 visibility 标签，也未把画面内可见当成无遮挡。详细日志见
  [TACO smoke](results/training_smoke_taco_20261029/)。
- 训练出的TACO checkpoint已通过原 `infer_marker_video.py` 完整推理路径：3帧、
  原生195点、778 MANO加密及overlay视频均成功导出；`validate_output.py`报告ok=true，
  `hand_778_exported=true`。结果见
  [完整推理检查](results/trained_checkpoint_inference_20261029/)。GPU进程已释放。
- tac2 GPU0/1 的ZeRO-2双卡验收通过：TACO第1步→保存分片→新torchrun恢复→STERA第2步。
  两rank裁剪前gradient norm均为2453.5842、98.42556，均有限；有效batch=2。
  独立进程加载portable inference checkpoint的636个状态项，在TACO和STERA各前向一次。
  总入口exit_code=0、training state=completed、最后GPU进程清单为空。记录在
  [双卡训练/恢复](results/training_zero2_smoke_20261029/)。两个dataset各只有一个3帧
  clip，因此同一步各rank使用相同样本；这验证通信/恢复，不验证数据多样性或泛化。
- 8卡、每卡8次梯度累积、有效batch64的两步测试也完成；第1步后退出torchrun，
  第2步由新进程从分片恢复。每步混合TACO/scene监督，裁剪前gradient norm分别
  924.0079、2301.366，独立推理回载636项，入口exit_code=0，GPU进程已释放。
  见 [8卡batch64结果](results/training_zero2_batch64_20261029/)。同样仅使用上述两个
  3帧clip，硬件为H100而非论文H20，不代表全数据训练、长窗口容量或论文数值复现。

## ZeRO-2 入口

```bash
torchrun --standalone --nproc_per_node=8 scripts/pwm/train_zero2.py \
  --config configs/pwm_baseline.toml --recipe configs/training_reconstruction_example.json \
  --index /path/index.jsonl --output /path/zero2_run
```

省略`--initialize`为随机初始化可训练模块；加官方marker checkpoint为posttraining。
`--resume`从输出目录`sharded/`恢复DeepSpeed模型/optimizer/step及所有rank的RNG。
world size必须一致。`inference.pt`只供推理，不能拿它冒充完整的分布式训练恢复点。
每卡累积8个clip、8卡有效batch64；采用microbatch1是为兼容不同数据集的标签集合，
并非声称逐bit复制作者的batch组织。实测不同rank分别用scene和hand监督时，首步在
优化器阶段阻塞，因此不支持跨rank任意混用监督图。当前采用确定性的dataset-uniform
调度：同一microstep所有rank使用同一个dataset/stride，各rank在该dataset内独立抽clip；
前向前检查图像shape一致，反向前检查有效loss项一致，不匹配则明确报错。
数据集权重均匀也是重建选择，不能视为论文公开参数。该策略写入checkpoint恢复契约。
已验证双卡跨step混合，以及8卡同一步内累积不同dataset的短测；全量训练尚未执行。

运行采用DeepSpeed 0.19.8的`torch_autocast` bf16模式，FP32参数/optimizer及显式FP32
几何运算得以保留，避免引擎禁用外部autocast引起DINO卷积dtype冲突。该版本初始化
广播还需要`zero_compat.py`处理原模型的零维gate，广播结束立即恢复原shape。修改
不改变checkpoint topology；它属于运行兼容实现，而非论文额外网络层。

复现分支 `pwm/training-reconstruction`，未合并到现有推理main。远程独立工作树：
`/home-mixed/liwenhao666/wm/PWM-training-reconstruction`。
外层 `codex-local-cuihu-tac2:pwm-training`；内层短测使用 `pwm-training-smoke`、
`pwm-training-taco-smoke`、`pwm-trained-inference-smoke`、`pwm-zero2-smoke`、
`pwm-zero2-batch64`，结束后退出。
原始运行产物位于
`/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke*_20261029`。
双卡运行目录为同一outputs下的 `training_zero2_smoke_20261029/`；分布式续训读取
`run/sharded/`，便携推理权重为 `run/inference.pt`。
8卡运行目录为同一outputs下的 `training_zero2_batch64_20261029/`，子目录契约相同。
本地操作日志 `/home/zy/server_logs/2026-10-29/codex-local-cuihu-tac2/`。

## 100步实验的就绪检查（2026-10-29）

**100步实验尚未启动。** 此处记录准备工作，不是新的训练结果。
再次检查上游HEAD仍为 `264f4df9fd42536aae4dba982d18272cc3334077`，只有main分支；
[训练代码询问](https://github.com/MINT-SJTU/EgoFound3R/issues/1)尚无回复。
[模型卡](https://huggingface.co/Frank-F2021/EgoFound3R)把`EgoFound3R.pt`定义为发布的推理
权重，并未推荐它作为复现原训练的初始化。可以确定的是加载预训练VGGT-Ω并冻结；
其余模块采用公开builder默认初始化属于我们的重建选择，尚无作者完整初始化配方佐证。

tac2当前八张H100 80GB均空闲；原始数据目录找到六集，HOI4D在检查的候选路径仍未找到。
六集目录存在不代表训练输入就绪。详细读盘检查见
[就绪审计](results/benchmark100_readiness_20261029/)。控制窗口使用
`codex-local-cuihu-tac2:pwm-benchmark`，只读检查内层为`pwm-benchmark100-readiness`。
进一步检查HOT3D Aria的199个序列目录，其中136个有非空MANO标注，所有检查目录的
直接子文件均没有`.vrs/.mp4/.avi/.mkv`；这不排除RGB存放在其他路径，但目前不能确认
可用的HOT3D训练视频。HOI4D额外检查`dataset/ego/HOI4D`等候选也没有找到。

代码已加入：

- `preflight_benchmark_data.py`：检查显式数据集覆盖、固定帧数/256²、版本化adapter、
  文件checksum、训练与验证episode隔离、内容不重复。默认训练七集、验证六集（不含STERA）；
  每集至少8个不同训练clip是本次工程测速的最低要求，不是论文公布参数。
- `train_zero2.py --save-initial`：保存第0步可训练模块，供相同验证集前后对比；记录
  初始化来源、实际输入clip/stride/帧数、每卡显存峰值和每步同步耗时。
- `timing.json`：区分初始化、训练step总时间、checkpoint开销和训练阶段wall time，
  给出剔除前5步后的mean/median/p95及3200步时间外推。尚无正式100步测量值。
- 正式测速应保留recipe.steps=3200、warmup_steps=100，通过`--stop-after 100`停止；
  不能把总schedule改成100步，也不能把重复两个3帧样本的结果解释成七集正式训练。

公开资料仍缺训练clip长度、stride配比、精确split、完整数据处理细节和部分损失系数。
任何实跑都须明确这些选择，以及数据范围是七集混合还是两集子集，不能声称完全照搬作者流程。
