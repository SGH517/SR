import torch.optim as optim
import torch.nn as nn
# 尝试导入 ultralytics 的 Detect 模块类型，如果不存在则忽略
try:
    from ultralytics.nn.modules.head import Detect
except ImportError:
    Detect = None
    print("Warning: Could not import ultralytics.nn.modules.head.Detect. Detector head grouping might be less precise.")


def get_optimizer_with_differential_lr(model, config):
    """
    根据配置为模型的不同部分设置差分学习率。

    Args:
        model (torch.nn.Module): 包含 ConditionalSR 和 Detector 的模型。
                                 需要有 'masker', 'sr_fast', 'sr_quality', 'detector' 属性。
        config (dict): 包含优化器和学习率设置的配置字典。
                       例如: config['optimizer'], config['learning_rates']

    Returns:
        torch.optim.Optimizer: 配置好的优化器。
    """
    lr_config = config['train']['learning_rates']
    opt_config = config['train']['optimizer']

    high_lr_params = []
    low_lr_params = []
    # other_params = [] # 用于未明确分组的参数

    # 检查模型组件是否存在
    has_masker = hasattr(model, 'masker') and model.masker is not None
    has_sr_fast = hasattr(model, 'sr_fast') and model.sr_fast is not None
    has_sr_quality = hasattr(model, 'sr_quality') and model.sr_quality is not None
    has_detector = hasattr(model, 'detector') and model.detector is not None and model.detector.model is not None

    print("--- Optimizer Parameter Grouping ---")

    # 1. Masker 参数 (通常高学习率)
    if has_masker:
        high_lr_params.extend(model.masker.parameters())
        print(f"Optimizer: Added {len(list(model.masker.parameters()))} Masker params to High LR group.")

    # 2. Detector 参数 (Backbone 低学习率, Head 高学习率)
    if has_detector:
        detector_backbone_params = []
        detector_head_params = []
        try:
            # 尝试通过模块类型识别检测头 (YOLOv8)
            if Detect is not None:
                for name, module in model.detector.model.named_modules():
                    if isinstance(module, Detect):
                        print(f"Optimizer: Identified Detector Head module: {name}")
                        for param in module.parameters():
                            detector_head_params.append(param)
                # 将剩余参数视为 Backbone
                all_detector_params = set(model.detector.model.parameters())
                head_param_set = set(detector_head_params)
                detector_backbone_params = list(all_detector_params - head_param_set)
            else:
                 # Fallback: 尝试根据名称或最后几层判断
                 all_params = list(model.detector.model.named_parameters())
                 # 启发式：假设最后几个模块是 head
                 num_total_params = len(all_params)
                 # 粗略估计 head 参数比例，例如最后 10% 或固定数量
                 num_head_params_fallback = max(1, num_total_params // 10) # 至少1个参数，或总数的10%
                 detector_head_params = [p for n, p in all_params[-num_head_params_fallback:]]
                 detector_backbone_params = [p for n, p in all_params[:-num_head_params_fallback]]
                 print(f"Optimizer: Fallback grouping for Detector. Added last {num_head_params_fallback} params to Head group.")


            if detector_head_params:
                 high_lr_params.extend(detector_head_params)
                 print(f"Optimizer: Added {len(detector_head_params)} Detector Head params to High LR group.")
            if detector_backbone_params:
                 low_lr_params.extend(detector_backbone_params)
                 print(f"Optimizer: Added {len(detector_backbone_params)} Detector Backbone params to Low LR group.")
            else:
                 print("Optimizer: No Detector Backbone params identified.")

        except Exception as e:
            print(f"Optimizer Warning: Error during Detector parameter grouping ({e}). Adding all detector params to Low LR group as fallback.")
            low_lr_params.extend(model.detector.model.parameters())
            print(f"Optimizer: Added {len(list(model.detector.model.parameters()))} Detector params to Low LR group (fallback).")


    # 3. SR 网络参数 (主体低学习率, 上采样层高学习率)
    def add_sr_params(sr_model, model_name):
        sr_high_lr = []
        sr_low_lr = []
        all_params = list(sr_model.named_parameters())

        # 识别上采样层参数
        upsample_params = []
        for name, module in sr_model.named_modules():
            # SRFast 的反卷积层
            if isinstance(module, nn.ConvTranspose2d):
                print(f"Optimizer: Identified {model_name} ConvTranspose2d module: {name}")
                for param in module.parameters():
                    upsample_params.append(param)
            # SRQuality 的 PixelShuffle 及其前面的卷积
            if isinstance(module, nn.PixelShuffle):
                 print(f"Optimizer: Identified {model_name} PixelShuffle module: {name}")
                 # 找到 PixelShuffle 前面的 Conv2d
                 parent_name = ".".join(name.split('.')[:-1]) # Get parent module name
                 parent_module = sr_model
                 for sub_name in parent_name.split('.'):
                     parent_module = getattr(parent_module, sub_name)
                 # Iterate through parent's children to find Conv2d before PixelShuffle
                 found_pixelshuffle = False
                 for child_name, child_module in parent_module.named_children():
                     if child_module is module:
                         found_pixelshuffle = True
                         break
                     if isinstance(child_module, nn.Conv2d):
                         print(f"Optimizer: Identified {model_name} Conv2d before PixelShuffle: {parent_name}.{child_name}")
                         for param in child_module.parameters():
                             upsample_params.append(param)
                 if not found_pixelshuffle:
                      print(f"Optimizer Warning: Could not find Conv2d before PixelShuffle in {model_name} at {name}.")


        upsample_param_set = set(upsample_params)
        for name, param in all_params:
            if param in upsample_param_set:
                sr_high_lr.append(param)
            else:
                sr_low_lr.append(param)

        # Fallback: 如果没有识别到特定上采样层，将最后几层放入高 LR
        if not sr_high_lr and len(all_params) > 2:
             print(f"Optimizer Warning: No specific upsample layers identified for {model_name}. Adding last 2 params to High LR group as fallback.")
             sr_high_lr.extend([p for n, p in all_params[-2:]])
             sr_low_lr = [p for n, p in all_params[:-2]]
        elif not sr_high_lr:
             print(f"Optimizer Warning: No specific upsample layers identified for {model_name} and not enough params for fallback. Adding all to Low LR group.")
             sr_low_lr.extend([p for n, p in all_params])


        high_lr_params.extend(sr_high_lr)
        low_lr_params.extend(sr_low_lr)
        print(f"Optimizer: Added {len(sr_high_lr)} {model_name} params to High LR group.")
        print(f"Optimizer: Added {len(sr_low_lr)} {model_name} params to Low LR group.")


    if has_sr_fast:
        add_sr_params(model.sr_fast, "SR_Fast")
    if has_sr_quality:
        add_sr_params(model.sr_quality, "SR_Quality")

    # Ensure no parameter is in both groups (shouldn't happen with extend/append if logic is correct)
    # And ensure all model parameters are included
    all_model_params = set(model.parameters())
    grouped_params = set(high_lr_params + low_lr_params)
    if len(all_model_params) != len(grouped_params):
        print(f"Optimizer Warning: Parameter count mismatch! Model has {len(all_model_params)} params, grouped has {len(grouped_params)}.")
        # Identify missing/duplicate parameters for debugging
        missing_params = list(all_model_params - grouped_params)
        if missing_params:
            print(f"Optimizer Warning: {len(missing_params)} parameters were not included in any group.")
            # Optionally add missing params to a default group or low_lr group
            low_lr_params.extend(missing_params)
            print(f"Optimizer: Added {len(missing_params)} missing params to Low LR group.")
        # Note: Finding duplicates is harder with lists, sets help check total count

    # Create parameter groups
    param_groups = [
        {'params': list(set(high_lr_params)), 'lr': lr_config['high_lr']}, # Use set to remove potential duplicates
        {'params': list(set(low_lr_params)), 'lr': lr_config['low_lr']},
        # {'params': other_params, 'lr': lr_config['default_lr']} # 如果有其他参数
    ]

    # Remove empty groups
    param_groups = [g for g in param_groups if g['params']]


    # 选择优化器
    optimizer_name = opt_config.get('name', 'AdamW').lower()
    optimizer_args = opt_config.get('args', {})

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(param_groups, **optimizer_args)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(param_groups, **optimizer_args)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(param_groups, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Optimizer: Created {optimizer_name} with differential learning rates:")
    for i, group in enumerate(optimizer.param_groups):
        group_type = "High LR" if group['lr'] == lr_config['high_lr'] else "Low LR"
        print(f"  Group {i} ({group_type}, {len(group['params'])} params): {group['lr']}")

    print("--- End Optimizer Parameter Grouping ---")

    return optimizer
