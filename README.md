# EvoDet

EvoDet 是一个基于 **YOLOv8** 的持续学习与域自适应目标检测实验框架。当前实现主要覆盖两条实验线：

1. **类别增量检测（Class-Incremental Object Detection）**
   - Finetune baseline
   - YOLO-LwF + OCDM replay memory
   - MilTech 2p2、COCO 40p10 等任务划分

2. **无监督域自适应检测（Unsupervised Domain Adaptation）**
   - YOLOv8 版 ConfMix
   - Source supervised pretrain
   - Source + Target ConfMix UDA
   - YOLOv8 DFL entropy uncertainty pseudo-labeling
   - 支持断点恢复

当前代码以 Ultralytics YOLOv8 为底座，保留 YOLO 官方数据格式、YOLO 风格 mAP 评估，并在此基础上扩展持续学习和域自适应逻辑。

---

## 1. 项目能力概览

### 1.1 类别增量检测

当前支持：

- 固定总类别数的 YOLOv8 检测头，例如 COCO 直接初始化为 80 类。
- 每个 task 只训练当前新类。
- 评估时评估所有 seen classes，即旧类 + 当前新类。
- task 0 无 teacher。
- task > 0 加载上一 task checkpoint 作为 teacher。
- 使用 LwF 蒸馏保持旧类输出。
- 使用 OCDM replay memory 保存和刷新代表性旧样本。
- 支持 task 边界和 epoch 级断点恢复。

典型 COCO 40p10 流程：

```text
task0: classes 0-39
task1: classes 40-49, seen 0-49
task2: classes 50-59, seen 0-59
task3: classes 60-69, seen 0-69
task4: classes 70-79, seen 0-79
```

当前实现采用：

```text
一开始即构建 80 类 YOLOv8 检测头；
每个 task 通过 dataset filter、loss mask、teacher、replay 控制训练类别；
不在 task1 重新创建 50 类 head，也不逐步扩展 head。
```

---

### 1.2 域自适应 ConfMix

当前支持：

- Stage 1：在 source/original domain 上监督预训练。
- Stage 2：加载 Stage 1 checkpoint，在 source + target 上做 ConfMix UDA。
- Stage 2 开始前，自动在 target val 上评估 source-only baseline。
- Target train 用于生成伪标签，不依赖真实标签参与 UDA loss。
- Target val 用于每个 epoch 后评估。
- 支持 source / uda / both 三种运行模式。
- 支持 source 阶段和 UDA 阶段断点恢复。
- YOLOv8 版本不修改 YOLOv8 head，而是用 DFL 分布熵估计 bbox uncertainty。

MilTech natural → dark 的逻辑：

```text
Stage 1:
  train: /datasets/MilTech/natural/images/train
  eval : /datasets/MilTech/natural/images/val

Stage 2:
  train source: /datasets/MilTech/natural/images/train  有标签
  train target: /datasets/MilTech/dark/images/train     伪标签
  eval target : /datasets/MilTech/dark/images/val
```

---

## 2. 推荐目录结构

```text
EvoDet/
├── clod_framework/
│   ├── data/
│   │   ├── builder.py
│   │   ├── uda_builder.py
│   │   ├── yolo_detection_dataset.py
│   │   ├── replay_pair_dataset.py
│   │   └── replay_yolo_dataset.py
│   ├── engine/
│   │   ├── evaluator.py
│   │   └── optim.py
│   ├── losses/
│   │   ├── detection_loss.py
│   │   └── yolo_lwf_replay_loss.py
│   ├── methods/
│   │   ├── finetune.py
│   │   ├── yolo_lwf_ocdm.py
│   │   └── confmix_yolov8.py
│   ├── models/
│   │   ├── base_detector.py
│   │   ├── builder.py
│   │   └── yolov8_detector.py
│   └── replay/
│       └── ocdm_memory.py
├── configs/
│   ├── datasets/
│   └── experiments/
├── scripts/
│   ├── train_incremental.py
│   └── train_uda.py
├── outputs/
├── weights/
├── requirements.txt
└── README.md
```

---

## 3. 环境准备

### 3.1 创建环境

示例环境名为 `yolov8`：

```bash
conda create -n yolov8 python=3.9 -y
conda activate yolov8
```

安装 PyTorch、Ultralytics、pyyaml、tqdm 等依赖：

```bash
pip install torch torchvision
pip install ultralytics pyyaml tqdm numpy opencv-python
```

如需使用 COCO 官方 API 评估，也可安装：

```bash
pip install pycocotools
```

当前主评估器为 YOLO 风格 evaluator，不强依赖 COCO JSON。

---

### 3.2 权重路径

`yolov8_detector.py` 不应写死 `/workspace/EvoDet` 或 `/workspace/Dvodet`。

推荐使用以下规则：

1. 如果配置里是绝对路径，直接使用。
2. 如果是相对路径，依次从以下位置查找：
   - 当前运行目录
   - 项目根目录
   - `项目根目录/weights`
   - `项目根目录/checkpoints`
3. 如果都找不到，交给 Ultralytics `YOLO(weight_name)` 尝试自动下载。
4. 如项目根目录不在默认位置，可以设置：

```bash
export EVODET_ROOT=/path/to/EvoDet
```

示例：

```yaml
model:
  pretrained: yolov8n.pt
  pretrained_type: detector
```

或：

```yaml
model:
  pretrained: /workspace/EvoDet/outputs/miltech_natural_source_pretrain_yolov8n/checkpoints/best.pt
  pretrained_type: detector
```

---

## 4. 数据格式

当前数据集建议使用 YOLO 官方检测格式：

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

COCO 转 YOLO 后一般是：

```text
/datasets/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

MilTech 示例：

```text
/datasets/MilTech/natural/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

/datasets/MilTech/dark/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每个 label 文件格式：

```text
class_id x_center y_center width height
```

坐标为归一化 YOLO 格式。

---

## 5. 数据集配置

### 5.1 MilTech natural

`configs/datasets/miltech_natural.yaml`

```yaml
name: miltech_natural
root: /datasets/MilTech/natural

train: images/train
val: images/val
test: images/test

num_classes: 4
nc: 4

names:
  - APC
  - Military Truck
  - Person
  - Tank
```

### 5.2 MilTech dark

`configs/datasets/miltech_dark.yaml`

```yaml
name: miltech_dark
root: /datasets/MilTech/dark

train: images/train
val: images/val
test: images/test

num_classes: 4
nc: 4

names:
  - APC
  - Military Truck
  - Person
  - Tank
```

---

## 6. 类别增量训练

入口：

```text
scripts/train_incremental.py
```

核心调用链：

```text
main()
  -> build_tasks()
  -> build_model()
  -> build_method()
  -> train_one_task()
      -> method.on_task_start()
      -> build_train_loader() or method.build_train_loader()
      -> method.training_step()
      -> evaluator.evaluate()
      -> save_task_model()
      -> method.on_task_end()
```

---


### 6.1 运行 MilTech 类增量

```bash
cd /workspace/EvoDet

python scripts/train_incremental.py \
  --config configs/experiments/miltech_2p2_yolo_lwf_ocdm.yaml
```

---

### 6.3 COCO 40p10

运行：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8s_yolo_lwf_ocdm.yaml
```

---

### 6.4 类增量输出

训练后会生成：

```text
outputs/<experiment_name>/
├── checkpoints/
│   ├── model_task_0.pt
│   ├── model_task_1.pt
│   ├── replay_memory.json
│   ├── replay_memory_task_0.json
│   └── training_state_latest.pt
├── metrics/
│   ├── mAPs_task_0.csv
│   ├── mAPs_task_1.csv
│   └── ocdm.csv
└── logs/
```

判断 task 是否完整完成，不应只看 `model_task_k.pt`，还应确认：

```text
model_task_k.pt + replay_memory_task_k.json
```

因为模型可能已经保存，但 replay memory 更新可能中断。

---

### 6.5 类增量恢复

普通从头训练：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8n_yolo_lwf_ocdm.yaml
```

自动恢复：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8n_yolo_lwf_ocdm.yaml \
  --resume
```

指定恢复文件：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8n_yolo_lwf_ocdm.yaml \
  --resume \
  --resume-path outputs/coco_40p10_yolov8n_yolo_lwf_ocdm/checkpoints/training_state_latest.pt
```

恢复状态包含：

```text
model_state_dict
optimizer_state_dict
scaler_state_dict
task_id
next_epoch
global_step
method_state_dict
replay memory
rng_state
config
```

---

## 7. YOLOv8 ConfMix 域自适应

入口：

```text
scripts/train_uda.py
```

支持三种模式：

```bash
--stage source
--stage uda
--stage both
```

其中：

```text
source: 只做 source supervised pretrain
uda   : 只做 ConfMix UDA
both  : 先 source pretrain，再自动加载 source best.pt 做 UDA
```

---

### 7.1 MilTech natural → dark

`configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml`

---

### 7.2 运行完整两阶段

```bash
cd /workspace/EvoDet

python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both
```

流程：

```text
Stage 1:
  natural train 监督训练
  natural val 评估
  保存 source best.pt

Stage 2:
  加载 source best.pt
  先在 dark val 上评估 source-only target baseline
  natural train + dark train 做 ConfMix UDA
  每个 epoch 在 dark val 上评估
```

---

### 7.3 跳过已完成的 source pretrain

如果 source pretrain 已经完成：

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both \
  --skip-source-if-exists
```

---

### 7.4 只运行第一阶段

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage source
```

---

### 7.5 只运行第二阶段

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage uda \
  --source-ckpt outputs/miltech_natural_source_pretrain_yolov8n/checkpoints/best.pt
```

---

### 7.6 UDA 断点恢复

自动恢复：

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both \
  --resume
```

指定恢复文件：

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage uda \
  --resume \
  --resume-path outputs/miltech_natural_to_dark_confmix_yolov8n/checkpoints/uda_training_state_latest.pt
```

恢复文件：

```text
outputs/<source_pretrain>/checkpoints/source_training_state_latest.pt
outputs/<uda_experiment>/checkpoints/uda_training_state_latest.pt
```

恢复状态包含：

```text
model_state_dict
optimizer_state_dict
scaler_state_dict
epoch
next_epoch
global_step
best_map
rng_state
config
source_checkpoint
```

---

## 8. YOLOv8 ConfMix 方法说明

原始 ConfMix 基于 YOLOv5，额外修改 YOLOv5 head 输出 bbox variance。当前实现基于 YOLOv8，不修改 YOLOv8 head，而是利用 YOLOv8 DFL 回归分布估计定位不确定性。

当前 YOLOv8 ConfMix 伪标签分数：

```text
det_conf       = max class probability
uncertainty    = normalized entropy of DFL distribution
certainty      = 1 - uncertainty
combined_conf  = det_conf * certainty
pseudo_score   = (1 - delta) * det_conf + delta * combined_conf
```

其中：

```text
delta = sigmoid_ramp(progress)
gamma = gamma_max * delta
```

总损失：

```text
total_loss = source_supervised_loss + lambda_mix * gamma * mixed_pseudo_loss
```

区域选择：

```text
将 target 图像划分为 4 个区域；
每个区域统计伪检测的 region score；
选择 region score 最高的 target 区域；
将该区域贴到 source 图像；
构造 mixed image 和 mixed pseudo labels；
用 mixed pseudo labels 计算 detection loss。
```

推荐配置：

```yaml
method:
  region_score_key: combined_conf
  uncertainty_power: 1.0
```

如果伪标签太少：

```yaml
method:
  pseudo_conf_thres: 0.15
  uncertainty_power: 0.5
```

如果伪标签噪声太大：

```yaml
method:
  pseudo_conf_thres: 0.35
  uncertainty_power: 1.5
```

---

## 9. 评估指标

当前 evaluator 使用 YOLO 官方风格指标：

```text
P
R
mAP50
mAP50-95
per-class AP50
per-class AP
```

评估流程：

```text
model prediction
→ NMS
→ 按类别和 IoU 匹配 GT
→ 统计 TP / FP / confidence / pred_cls / target_cls
→ ap_per_class
→ P / R / mAP50 / mAP50-95
```

类别增量时：

```text
task0 eval classes = task0 classes
task1 eval classes = task0 + task1 seen classes
task2 eval classes = task0 + task1 + task2 seen classes
...
```

UDA 时：

```text
Stage 1 eval: source val
Pre-UDA eval: target val, epoch=0
UDA eval: target val, epoch=1..N
```

---

## 10. 常见问题

### 10.1 为什么第二阶段显存没有明显增加？

YOLO-LwF+OCDM 第二阶段加载 teacher，但 teacher 通常：

```text
eval mode
requires_grad=False
torch.no_grad()
无 optimizer state
```

teacher 只增加少量参数显存，不保存反向传播图。训练显存主要来自 student activations、loss graph、optimizer state 和 batch tensor，所以显存看起来可能和第一阶段差不多。

---

### 10.2 为什么 task0 模型保存后不能直接进 task1？

因为 YOLO-LwF+OCDM 不只需要：

```text
model_task_0.pt
```

还需要：

```text
replay_memory_task_0.json
```

如果 task0 的 model 保存成功，但 memory update 失败，task0 不能算完整完成。恢复逻辑应同时检查模型和 replay memory。

---

### 10.3 为什么 pseudo label 阶段必须关闭 Mosaic？

因为 replay memory 保存的是原图路径和原图坐标下的伪标签。如果 pseudo label 生成时仍然使用 Mosaic/MixUp，生成的 box 对应的是增强图，不再对应原始图片路径，会污染 replay memory。

因此 pseudo label 阶段需要：

```text
augment=False
mosaic=0
mixup=0
cutmix=0
copy_paste=0
```

并在结束后恢复 dataset transforms。

---

### 10.4 为什么 OfficialYOLODetectionDataset 不能在 get_labels() 里过滤类别？

Ultralytics YOLODataset 假设：

```text
self.im_files[i] 对应 self.labels[i]
```

如果只在 `get_labels()` 里删除 label，而不同步删除 `im_files`、`label_files`、`npy_files`、缓存列表，就会导致图片和标签错位。

正确做法：

```text
先让 YOLODataset 正常初始化；
然后在 __init__ 后统一过滤 labels / im_files / label_files / npy_files / ims / im_hw0 / im_hw；
最后重建 buffer。
```

---

### 10.5 为什么 buffer 不能清空？

Ultralytics Mosaic 会从：

```text
dataset.buffer
```

抽取额外图片。如果 task filter 后写成：

```python
self.buffer = []
```

Mosaic 会报：

```text
IndexError: list index out of range
```

正确做法：

```python
self.buffer = list(range(self.ni))
```

---

### 10.6 COCO 类增量为什么 task1 数据量可能比 task0 大？

类别增量数据过滤通常按“图片中是否包含当前 task 类别”筛图，而不是按实例数平均。COCO 或 MilTech 图像可能同时包含旧类和新类，也可能新类更常见，因此不同 task 的 image count 不一定均衡。

---

### 10.7 COCO 用 mAP50 还是 mAP50-95？

VOC 通常看 mAP50。

COCO 推荐看：

```text
mAP50-95
```

当前 evaluator 会同时输出：

```text
mAP50
mAP50-95
```

---

## 11. 推荐调试顺序

### 11.1 类增量

先跑小数据集：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/miltech_2p2_yolo_lwf_ocdm.yaml
```

确认：

```text
task0 能训练和评估
task0 结束能更新 replay memory
task1 能加载 teacher
task1 loss 中 lwf 不为 0
task1 eval classes 包含旧类和新类
```

再跑 COCO：

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8n_yolo_lwf_ocdm.yaml
```

---

### 11.2 UDA

先把 epoch 改小：

```yaml
source_pretrain:
  epochs: 1

training:
  epochs: 1
```

运行：

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both
```

确认：

```text
Stage 1 natural source pretrain 正常
Stage 2 前有 Pre-UDA Eval
ConfMix 训练日志有 pseudo / gamma / unc / cert / comb
每个 UDA epoch 在 dark val 上评估
```

---

## 12. 关键文件阅读顺序

### 12.1 类增量

```text
scripts/train_incremental.py
clod_framework/methods/yolo_lwf_ocdm.py
clod_framework/losses/yolo_lwf_replay_loss.py
clod_framework/replay/ocdm_memory.py
clod_framework/data/replay_pair_dataset.py
clod_framework/data/replay_yolo_dataset.py
```

### 12.2 UDA ConfMix

```text
scripts/train_uda.py
clod_framework/data/uda_builder.py
clod_framework/methods/confmix_yolov8.py
clod_framework/losses/detection_loss.py
clod_framework/engine/evaluator.py
```

### 12.3 YOLOv8 适配层

```text
clod_framework/models/yolov8_detector.py
clod_framework/models/builder.py
clod_framework/data/yolo_detection_dataset.py
clod_framework/engine/optim.py
```

---

## 13. 最小心智模型

```text
Script
  负责训练流程、阶段、恢复、评估、保存。

Method
  负责算法逻辑：
    YOLO-LwF+OCDM: teacher / replay / LwF / memory update
    ConfMix: source supervised / target pseudo labels / region mixing

Model
  负责 YOLOv8 DetectionModel 构建、权重加载、forward、predict_raw。

Dataset
  负责 YOLO 官方数据加载、task class filter、UDA source/target loader。

Loss
  负责 YOLOv8 detection loss、LwF distillation、replay loss、ConfMix mixed loss。

Evaluator
  负责 YOLO 风格 P/R/mAP50/mAP50-95。
```

---

## 14. 快速命令汇总

### MilTech 类增量

```bash
python scripts/train_incremental.py \
  --config configs/experiments/miltech_2p2_yolo_lwf_ocdm.yaml
```

### COCO 40p10 类增量

```bash
python scripts/train_incremental.py \
  --config configs/experiments/coco_40p10_yolov8n_yolo_lwf_ocdm.yaml
```

### MilTech natural → dark UDA 两阶段

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both
```

### MilTech UDA 恢复

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage both \
  --resume
```

### 只跑 UDA 第二阶段

```bash
python scripts/train_uda.py \
  --config configs/experiments/miltech_natural_to_dark_confmix_yolov8.yaml \
  --stage uda \
  --source-ckpt outputs/miltech_natural_source_pretrain_yolov8n/checkpoints/best.pt
```

---

## 15. 当前实现边界

- YOLO-LwF+OCDM 当前采用固定总类别数 head，不做动态 head 扩展。
- ConfMix 当前是 YOLOv8 适配版，不修改 YOLOv8 head。
- YOLOv8 ConfMix 的 uncertainty 来自 DFL entropy，而不是 YOLOv5 原始 variance head。
- UDA 阶段 target train 默认不使用真实标签参与 loss，只用于生成伪标签。
- 当前 evaluator 是 YOLO 风格 evaluator；如需严格 COCO API，需要额外接入 COCO JSON 和 `pycocotools.COCOeval`。
=======