import torch.optim as optim

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
    lr_config = config['learning_rates']
    opt_config = config['optimizer']

    high_lr_params = []
    low_lr_params = []
    other_params = [] # 用于未明确分组的参数

    # 检查模型组件是否存在
    has_masker = hasattr(model, 'masker') and model.masker is not None
    has_sr_fast = hasattr(model, 'sr_fast') and model.sr_fast is not None
    has_sr_quality = hasattr(model, 'sr_quality') and model.sr_quality is not None
    has_detector = hasattr(model, 'detector') and model.detector is not None and model.detector.model is not None

    # 分组参数
    if hasattr(model, 'masker') and model.masker is not None:
        high_lr_params.extend(model.masker.parameters())
    if has_masker:
        high_lr_params.extend(model.masker.parameters())
        print("Optimizer: Added Masker params to High LR group.")

    if has_detector:
        # 假设 YOLO 模型有 head 和 backbone/body 属性或类似结构
        # 这需要根据具体 YOLO 实现调整
        try:
            # 尝试区分 head 和 backbone (ultralytics YOLOv8 可能没有直接的 .head/.backbone)
            # 临时策略：将所有 detector 参数放入 low_lr_params，然后在下面覆盖 head
            # 或者，更精细地，根据模块名称判断
            detector_backbone_params = []
            detector_head_params = []
            for name, param in model.detector.model.named_parameters():
                 # 示例：假设检测头相关的层名包含 'detect' 或 'head'
                 if 'detect' in name or 'head' in name: # 需要根据实际模型调整
                     detector_head_params.append(param)
                 else:
                     detector_backbone_params.append(param)

            if detector_head_params:
                 high_lr_params.extend(detector_head_params)
                 print(f"Optimizer: Added {len(detector_head_params)} Detector Head params to High LR group.")
            if detector_backbone_params:
                 low_lr_params.extend(detector_backbone_params)
                 print(f"Optimizer: Added {len(detector_backbone_params)} Detector Backbone params to Low LR group.")
            else: # 如果无法区分，将所有 detector 参数放入 low LR 组
                 print("Optimizer: Could not clearly distinguish Detector head/backbone. Adding all Detector params to Low LR group.")
                 low_lr_params.extend(model.detector.model.parameters())

        except AttributeError as e:
            print(f"Optimizer Warning: Could not access specific parts of the detector model ({e}). Adding all detector params to Low LR group.")
            low_lr_params.extend(model.detector.model.parameters())

    def add_sr_params(sr_model, model_name):
        # 区分 SR 模型的最后几层和其余层
        sr_high_lr = []
        sr_low_lr = []
        num_params = len(list(sr_model.parameters()))
        # 临时策略：将最后 1/4 的参数视为高学习率 (需要更精细的层选择)
        # 或者根据层名判断 (如 'upsample', 'tail')
        last_layer_names = ['upsample', 'deconvolution', 'tail'] # 示例
        all_params = list(sr_model.named_parameters())
        high_lr_added = False
        for name, param in reversed(all_params):
            if any(layer_name in name for layer_name in last_layer_names) and not high_lr_added:
                 sr_high_lr.append(param)
                 high_lr_added = True # 假设最后匹配的层组是高 LR
            else:
                 sr_low_lr.append(param)
        # 如果没有匹配到特定层名，将最后几层放入高 LR
        if not high_lr_added and len(all_params) > 2:
             sr_high_lr.extend([p for n, p in all_params[-2:]]) # 最后两个参数
             sr_low_lr = [p for n, p in all_params[:-2]]
             print(f"Optimizer: Adding last 2 params of {model_name} to High LR group (fallback).")
        elif not high_lr_added:
             sr_low_lr.extend([p for n, p in all_params]) # 全部放入低 LR
             print(f"Optimizer: Could not identify last layers of {model_name}. Adding all to Low LR group.")
        else:
             print(f"Optimizer: Added last layers of {model_name} to High LR group based on name.")

        high_lr_params.extend(sr_high_lr)
        low_lr_params.extend(sr_low_lr)
        print(f"Optimizer: Added {len(sr_high_lr)} {model_name} params to High LR group.")
        print(f"Optimizer: Added {len(sr_low_lr)} {model_name} params to Low LR group.")


    if has_sr_fast:
        add_sr_params(model.sr_fast, "SR_Fast")
    if has_sr_quality:
        add_sr_params(model.sr_quality, "SR_Quality")

    # 创建参数组
    param_groups = [
        {'params': high_lr_params, 'lr': lr_config['high_lr']},
        {'params': low_lr_params, 'lr': lr_config['low_lr']},
        # {'params': other_params, 'lr': lr_config['default_lr']} # 如果有其他参数
    ]

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
    print(f"  High LR group ({len(high_lr_params)} params): {lr_config['high_lr']}")
    print(f"  Low LR group ({len(low_lr_params)} params): {lr_config['low_lr']}")

    return optimizer
