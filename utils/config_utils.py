# utils/config_utils.py
import os
import logging
from typing import Dict, Optional

# 获取一个 logger 实例
# 如果您的项目中有统一的 logger 设置，可以考虑从那里导入
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def _check_path_exists(path: Optional[str], description: str, logger_instance: logging.Logger) -> bool:
    """辅助函数，检查路径是否存在并记录错误。"""
    if not path or not os.path.exists(path):
        logger_instance.error(f"{description} 未找到或路径无效: {path}")
        return False
    return True

def _check_key_exists(config_dict: Dict, key: str, description: str, logger_instance: logging.Logger) -> bool:
    """辅助函数，检查配置字典中是否存在某个键。"""
    if key not in config_dict:
        logger_instance.error(f"配置中缺少 '{description}' (键: {key})")
        return False
    return True

def _validate_sr_model_config(model_cfg: Dict, model_name: str, dataset_scale_factor: int, logger_instance: logging.Logger) -> bool:
    """校验单个SR模型的配置。"""
    is_valid = True
    if not _check_key_exists(model_cfg, 'scale_factor', f"{model_name}的缩放因子", logger_instance):
        is_valid = False
    elif model_cfg.get('scale_factor') != dataset_scale_factor:
        logger_instance.error(
            f"{model_name}的缩放因子 ({model_cfg.get('scale_factor')}) 与数据集的缩放因子 ({dataset_scale_factor}) 不匹配。"
        )
        is_valid = False

    if not _check_key_exists(model_cfg, 'output_path', f"{model_name}的输出路径", logger_instance):
        # 对于 stage2，output_path 是必须的
        # 对于 stage3，ConditionalSR内部加载权重，可能不需要外部output_path，但模型结构参数仍需校验
        pass # 根据具体阶段的严格程度决定是否将 is_valid 设为 False

    # 可以添加更多针对 SR 模型参数的校验，如 in_channels, d, s, m 等类型或范围检查
    return is_valid

def _validate_masker_config(masker_cfg: Dict, dataset_scale_factor: int, logger_instance: logging.Logger) -> bool:
    """校验Masker模型的配置。"""
    is_valid = True
    if not _check_key_exists(masker_cfg, 'output_patch_size', "Masker的输出块大小", logger_instance):
        is_valid = False
    else:
        patch_size = masker_cfg.get('output_patch_size')
        if not isinstance(patch_size, int) or patch_size <= 0:
            logger_instance.error(f"Masker的 output_patch_size ({patch_size}) 必须是正整数。")
            is_valid = False
        elif dataset_scale_factor % patch_size != 0 and logger_instance: # 确保 logger 实例存在
            logger_instance.warning(
                f"数据集的缩放因子 ({dataset_scale_factor}) 不能被 Masker的 output_patch_size ({patch_size}) 整除，"
                "这可能导致掩码对齐问题。"
            )
    if not _check_key_exists(masker_cfg, 'threshold', "Masker的阈值", logger_instance):
        is_valid = False
    else:
        threshold = masker_cfg.get('threshold')
        if not isinstance(threshold, (float, int)) or not (0 <= threshold <= 1):
            logger_instance.warning(f"Masker的阈值 ({threshold}) 超出了期望的 [0, 1] 范围。")
    return is_valid


def validate_config(config: Dict, stage_name: str, logger_instance: Optional[logging.Logger] = None) -> bool:
    """
    校验给定训练阶段的配置。

    参数:
        config (dict): 加载的配置字典。
        stage_name (str): 阶段名称 ("stage1_yolo", "stage2_sr", "stage3_joint")。
        logger_instance (Optional[logging.Logger]): 用于记录日志的 logger 实例。

    返回:
        bool: 如果配置有效则为 True，否则为 False。
    """
    log = logger_instance if logger_instance else logger # 使用传入的logger或模块级logger
    log.info(f"--- 开始校验配置: {stage_name} ---")
    is_valid = True

    dataset_config = config.get('dataset', {})
    model_config = config.get('model', {})
    train_config = config.get('train', {})

    if not dataset_config:
        log.error("配置中缺少 'dataset' 部分。")
        is_valid = False
    if not model_config and stage_name != "stage2_sr": # stage2 的模型配置在 models 下
        log.error("配置中缺少 'model' 部分。")
        is_valid = False
    if not train_config:
        log.error("配置中缺少 'train' 部分。")
        is_valid = False

    if not is_valid: # 如果基本部分缺失，则提前返回
        log.error(f"--- 配置校验失败 (基本结构错误): {stage_name} ---")
        return False

    # --- Dataset Config Validation ---
    if not _check_key_exists(dataset_config, 'scale_factor', "数据集缩放因子", log):
        is_valid = False
    else:
        scale_factor = dataset_config.get('scale_factor')
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            log.error(f"数据集的 scale_factor ({scale_factor}) 必须是正整数。")
            is_valid = False

    if stage_name == "stage1_yolo":
        if not _check_path_exists(dataset_config.get('train_image_dir'), "训练图像目录", log):
            is_valid = False
        # val_image_dir 的检查可以根据 args.enable_eval 决定是否严格
        if config.get('args', {}).get('enable_eval', False): # 假设 args 会被合并到 config 中
            if not _check_path_exists(dataset_config.get('val_image_dir'), "验证图像目录", log):
                log.warning("启用了评估，但验证图像目录未找到或无效。") # 改为警告，因为YOLO脚本可能会处理
                # is_valid = False # 可以选择是否因此而使配置无效

    elif stage_name == "stage2_sr":
        if not _check_path_exists(dataset_config.get('base_dir'), "数据集基础目录", log):
            is_valid = False
        sr_models_config = config.get('models', {}) # stage2 的模型配置在 'models' 下
        if not sr_models_config:
            log.error("配置中缺少 'models' 部分 (针对 stage2_sr)。")
            is_valid = False
        else:
            sf = dataset_config.get('scale_factor', -1) # 获取前面校验过的 sf
            if not _validate_sr_model_config(sr_models_config.get('sr_fast', {}), "SR_Fast", sf, log):
                is_valid = False
            if not _validate_sr_model_config(sr_models_config.get('sr_quality', {}), "SR_Quality", sf, log):
                is_valid = False
        # 校验 evaluation 部分 (如果存在)
        eval_cfg = train_config.get('evaluation', {})
        if eval_cfg:
            val_dataset_cfg = eval_cfg.get('val_dataset', {})
            if val_dataset_cfg:
                if not _check_path_exists(val_dataset_cfg.get('base_dir'), "验证数据集基础目录", log):
                    log.warning("配置了验证，但验证数据集基础目录未找到或无效。")


    elif stage_name == "stage3_joint":
        if not _check_path_exists(dataset_config.get('image_dir'), "图像目录", log):
            is_valid = False
        if not _check_path_exists(dataset_config.get('annotation_file'), "标注文件", log):
            is_valid = False

        sf = dataset_config.get('scale_factor', -1)
        if not _validate_sr_model_config(model_config.get('sr_fast', {}), "SR_Fast", sf, log):
            is_valid = False
        if not _validate_sr_model_config(model_config.get('sr_quality', {}), "SR_Quality", sf, log):
            is_valid = False
        if not _validate_masker_config(model_config.get('masker', {}), sf, log):
            is_valid = False

        weights_cfg = model_config.get('weights', {})
        required_weights = {'detector': "检测器权重", 'sr_fast': "SR_Fast权重", 'sr_quality': "SR_Quality权重"}
        for key, desc in required_weights.items():
            if not _check_path_exists(weights_cfg.get(key), desc, log):
                is_valid = False
        if weights_cfg.get('masker') and not os.path.exists(weights_cfg['masker']):
            log.warning(f"提供了Masker权重路径，但文件未找到: {weights_cfg['masker']}")


        if not _check_key_exists(model_config, 'num_classes', "模型类别数 (num_classes)", log): is_valid = False
        yolo_params_cfg = model_config.get('yolo_params', {})
        if not _check_key_exists(yolo_params_cfg, 'reg_max', "YOLO参数 reg_max", log): is_valid = False
        if not _check_key_exists(yolo_params_cfg, 'strides', "YOLO参数 strides", log): is_valid = False
        if not train_config.get('yolo_hyp'):
            log.warning("配置中缺少 'train.yolo_hyp'。如果损失函数需要，将使用默认增益。")

        # 校验联合损失权重
        loss_weights_cfg = train_config.get('loss_weights', {})
        if not isinstance(loss_weights_cfg.get('detection'), (float, int)):
            log.error("train.loss_weights.detection 必须是数字。")
            is_valid = False
        if not isinstance(loss_weights_cfg.get('sparsity'), (float, int)):
            log.error("train.loss_weights.sparsity 必须是数字。")
            is_valid = False
        # 可以为其他损失权重添加类似检查

    else:
        log.warning(f"未知的校验阶段名称: {stage_name}")
        is_valid = False

    if is_valid:
        log.info(f"--- 配置校验通过: {stage_name} ---")
    else:
        log.error(f"--- 配置校验失败: {stage_name} ---")
    return is_valid