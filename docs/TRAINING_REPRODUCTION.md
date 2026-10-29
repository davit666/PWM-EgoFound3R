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
| 分布式 | 8 H20、8/device、ZeRO-2、effective64 | **未实现/未验收**；当前单设备 batch8 不等价 |
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
architecture contract、optimizer、step、Python/Torch/CUDA RNG、recipe、index hash。
未带内容 hash 的 index 无法防止 NPZ 被外部替换。`--stop-after` 可用于恢复测试且不改 LR
schedule。保存采用临时文件原子替换。当前没有分布式状态恢复保证。

## 数据与评测验收门槛

原始训练集：H2O、HOT3D Aria、OakInk-v2、TACO、HOI4D、ARCTIC、STERA-10M。
前六集 eval windows 数分别283、400、400、400、461、434，共2378；STERA仅训练。
训练 sequences 数分别138、136、565、839、1431、226，STERA all。作者未提供具体 id。
重新划分只能注明为我们自己的复现划分，不能直接与论文表格声称相同测试集。

2026-10-29 在 tac2 发现旧表六个 `/share/project/...` 路径不可见，正在复核其他挂载。
目录存在、文件数/大小验收、标签 schema 可用、转换完成、训练验证通过是不同状态。
原始标注审计见 `docs/results/training_source_audit_20261029/`，这是有界抽样而非全量验收。

完整完成条件：7集 adapter 实测、固定可追溯 split、所有可用监督接通、实际完整模型
backward/恢复/推理回载、8卡ZeRO-2训练3200步、正式2378窗口评测及与论文差异分析。
在这些条件满足前，只报告具体通过的检查和实际训练步数。
