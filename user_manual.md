# 动态超分辨率目标检测系统使用手册

## 1. 系统概述
本系统实现了一个基于条件掩码的动态超分辨率目标检测网络，包含三个阶段：
1. YOLO检测器预训练
2. SR_Fast和SR_Quality超分辨率网络预训练 
3. 联合微调

## 2. 项目结构
```
.
├── configs/                  # 配置文件
│   ├── stage1_yolo_pretrain.yaml   # 阶段1配置
│   ├── stage2_sr_pretrain.yaml     # 阶段2配置
│   └── stage3_joint_finetune.yaml  # 阶段3配置
├── data/                      # 数据处理脚本
│   ├── prepare_detection_data.py  # 准备检测数据
│   └── prepare_sr_data.py        # 准备超分辨率数据
├── models/                    # 模型实现
│   ├── conditional_sr.py      # 条件超分辨率主模型
│   ├── detector.py            # YOLO检测器封装
│   ├── masker.py              # 掩码生成器
│   ├── sr_fast.py             # 快速超分辨率网络
│   ├── sr_quality.py          # 高质量超分辨率网络
│   └── ...                    # 其他模型文件
├── utils/                     # 工具函数
│   ├── logger.py              # 日志工具
│   ├── gumbel.py              # Gumbel采样工具
│   ├── flops_counter.py       # FLOPs计算工具
│   ├── evaluation_utils.py    # 评估工具
│   └── ...                    # 其他工具文件
```

## 3. 环境配置
```bash
pip install -r requirements.txt
```

## 4. 训练流程

### 4.1 数据准备
```bash
# 准备超分辨率数据
python data/prepare_sr_data.py --input_dir [HR_IMAGES_DIR] --output_dir [OUTPUT_DIR] --scale_factor 4

# 准备检测数据 
python data/prepare_detection_data.py --input_dir [HR_IMAGES_DIR] --annotation_file [ANNOTATION_FILE] --output_dir [OUTPUT_DIR] --scale_factor 4
```

### 4.2 阶段训练
```bash
# 阶段1: YOLO预训练
python stage1_pretrain_yolo.py --config configs/stage1_yolo_pretrain.yaml

# 阶段2: SR网络预训练
python stage2_pretrain_sr.py --config configs/stage2_sr_pretrain.yaml

# 阶段3: 联合微调
python stage3_finetune_joint.py --config configs/stage3_joint_finetune.yaml
```

## 5. 模型推理
```python
from models.conditional_sr import ConditionalSR
import torch

model = ConditionalSR.from_pretrained("checkpoints/joint_best.pth")
lr_image = load_image("test.jpg")  # [1,3,H,W] tensor
results = model(lr_image)

# 获取输出
sr_image = results["sr_image"]  # 超分辨率结果
detections = results["detections"]  # 检测结果(BBox列表)
mask = results["mask_coarse"]  # 条件掩码
```

## 6. 评估指标
```bash
python evaluate.py --config configs/stage3_joint_finetune.yaml
```
评估指标包括：
- 目标检测:mAP50, mAP50-95
- 超分辨率:PSNR, SSIM
- 计算效率:FLOPs, 参数量
- 动态执行:掩码稀疏度

## 7. 配置文件说明
主要配置参数：
```yaml
# stage3_joint_finetune.yaml示例
model:
  weights:
    detector: "checkpoints/yolo_pretrained_hr.pth"
    sr_fast: "checkpoints/sr_fast.pth"
    sr_quality: "checkpoints/sr_quality.pth"
    masker: "checkpoints/masker.pth"
  
train:
  batch_size: 16
  epochs: 100
  gumbel:
    initial_tau: 1.0  # Gumbel温度初始值
    final_tau: 0.1   # Gumbel温度最终值
    anneal_epochs: 50 # 退火周期数
```

## 8. 结果可视化
使用TensorBoard查看训练过程:
```bash
tensorboard --logdir logs/
```
