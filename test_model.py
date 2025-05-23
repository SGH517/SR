# test_model.py
import torch
import argparse
import logging # 导入 logging

# 从 models 导入
from models.conditional_sr import ConditionalSR
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker

# 从 utils 导入
from utils.common_utils import get_device

# 设置一个简单的 logger
logger_test = logging.getLogger(__name__)
if not logger_test.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 该脚本用于测试模型的初始化和前向传播。
def parse_args():
    parser = argparse.ArgumentParser(description="测试模型初始化和前向传播")
    parser.add_argument("--use_gpu", action="store_true", help="如果可用，使用 GPU 进行测试")
    # 以下参数在原始脚本中存在，但在此测试脚本的当前逻辑下可能不直接使用，保留以备将来扩展
    # parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for faster GPU transfer")
    # parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别。")
    return parser.parse_args()

def test_model_initialization_and_forward():
    args = parse_args()

    # 设置日志级别
    try:
        logger_test.setLevel(args.log_level.upper())
    except ValueError:
        logger_test.setLevel(logging.INFO)
        logger_test.warning(f"无效的日志级别 '{args.log_level}'。使用 INFO 级别。")

    # 设备选择
    device = get_device(args.use_gpu, logger_test)
    logger_test.info(f"测试将在设备: {device} 上运行")

    # --- 测试 SRFast 初始化 ---
    try:
        sr_fast = SRFast(scale_factor=4).to(device)
        logger_test.info("SRFast 模型初始化成功。")
        dummy_input_srfast = torch.randn(1, 3, 32, 32).to(device) # 使用较小尺寸测试
        output_srfast = sr_fast(dummy_input_srfast)
        logger_test.info(f"SRFast 前向传播成功。输入形状: {dummy_input_srfast.shape}, 输出形状: {output_srfast.shape}")
        assert output_srfast.shape == (1, 3, 32 * 4, 32 * 4), "SRFast 输出形状不匹配"
    except Exception as e:
        logger_test.error(f"SRFast 初始化或前向传播失败: {e}", exc_info=True)
        return # 如果基础组件失败，后续测试可能无意义

    # --- 测试 SRQuality 初始化 ---
    try:
        sr_quality = SRQuality(scale_factor=4).to(device)
        logger_test.info("SRQuality 模型初始化成功。")
        dummy_input_srquality = torch.randn(1, 3, 32, 32).to(device)
        output_srquality = sr_quality(dummy_input_srquality)
        logger_test.info(f"SRQuality 前向传播成功。输入形状: {dummy_input_srquality.shape}, 输出形状: {output_srquality.shape}")
        assert output_srquality.shape == (1, 3, 32 * 4, 32 * 4), "SRQuality 输出形状不匹配"
    except Exception as e:
        logger_test.error(f"SRQuality 初始化或前向传播失败: {e}", exc_info=True)
        return

    # --- 测试 Masker 初始化 ---
    try:
        # Masker 参数需要与 ConditionalSR 中期望的一致，或者提供一个最小配置
        masker = Masker(in_channels=3, base_channels=32, num_blocks=4, output_patch_size=16).to(device)
        logger_test.info("Masker 模型初始化成功。")
        dummy_input_masker = torch.randn(1, 3, 64, 64).to(device) # Masker 输入 LR 尺寸
        output_masker = masker(dummy_input_masker)
        logger_test.info(f"Masker 前向传播成功。输入形状: {dummy_input_masker.shape}, 输出形状: {output_masker.shape}")
        # 期望输出: (B, 1, H_lr / patch_size, W_lr / patch_size)
        expected_mask_h = 64 // 16
        expected_mask_w = 64 // 16
        assert output_masker.shape == (1, 1, expected_mask_h, expected_mask_w), "Masker 输出形状不匹配"
    except Exception as e:
        logger_test.error(f"Masker 初始化或前向传播失败: {e}", exc_info=True)
        return

    # --- 测试 ConditionalSR 初始化 ---
    # ConditionalSR 的初始化现在需要一个 config 对象
    # 我们提供一个最小化的 mock config
    mock_conditional_sr_config = {
        'model': {
            'masker': {
                'threshold': 0.5 # ConditionalSR._validate_config 可能需要
            },
            'weights': { # 确保 weights 键存在，即使为空
                'detector': None, # 推理/测试时，检测器权重可以不提供，或者提供一个假路径（如果需要测试加载逻辑）
            }
            # 可以根据需要添加 sr_fast, sr_quality 的配置，但 ConditionalSR 的 __init__ 现在接收实例
        },
        'train': {} # _validate_config 可能检查 'train'
    }
    try:
        # 注意：ConditionalSR 初始化时，detector_weights 可以为 None 或无效路径，
        # 它内部会处理这种情况并使 self.detector 为 None。
        # sr_fast_weights, sr_quality_weights, masker_weights 也可为 None。
        conditional_sr_model = ConditionalSR(
            sr_fast=sr_fast, # 使用上面已初始化的实例
            sr_quality=sr_quality, # 使用上面已初始化的实例
            masker=masker, # 使用上面已初始化的实例
            detector_weights=None, # 测试时不加载实际的检测器权重，除非有专门的测试权重
            sr_fast_weights=None,  # 不加载预训练权重，测试纯粹的模块结构和前向逻辑
            sr_quality_weights=None,
            masker_weights=None,
            device=str(device),
            config=mock_conditional_sr_config
        ).to(device)
        logger_test.info("ConditionalSR 模型初始化成功。")
    except Exception as e:
        logger_test.error(f"ConditionalSR 初始化失败: {e}", exc_info=True)
        return

    # --- 测试 ConditionalSR 前向传播 ---
    dummy_input_conditional_sr = torch.randn(1, 3, 64, 64).to(device) # LR input
    try:
        # 测试训练模式下的前向传播 (Gumbel mask, no hard_mask_inference)
        logger_test.info("测试 ConditionalSR 训练模式前向传播...")
        conditional_sr_model.train()
        output_train = conditional_sr_model(dummy_input_conditional_sr, temperature=1.0)
        logger_test.info(f"ConditionalSR 训练模式前向传播成功。输出键: {list(output_train.keys())}")
        assert "sr_image" in output_train and output_train["sr_image"] is not None
        assert output_train["sr_image"].shape == (1, 3, 64 * 4, 64 * 4), "ConditionalSR 训练模式 sr_image 输出形状不匹配"
        assert "mask_coarse" in output_train and output_train["mask_coarse"] is not None
        assert output_train["mask_coarse"].shape == (1, 1, 64 // 16, 64 // 16), "ConditionalSR 训练模式 mask_coarse 输出形状不匹配"
        assert "mask_fused" in output_train and output_train["mask_fused"] is not None
        assert output_train["mask_fused"].shape == (1, 1, 64 * 4, 64 * 4), "ConditionalSR 训练模式 mask_fused 输出形状不匹配"
        logger_test.info(f"  SR 图像形状: {output_train['sr_image'].shape}, 粗粒度掩码形状: {output_train['mask_coarse'].shape}, 融合掩码形状: {output_train['mask_fused'].shape}")


        # 测试推理模式下的前向传播 (使用硬掩码)
        logger_test.info("测试 ConditionalSR 推理模式 (hard_mask=True) 前向传播...")
        conditional_sr_model.eval()
        output_eval_hard = conditional_sr_model(dummy_input_conditional_sr, hard_mask_inference=True)
        logger_test.info(f"ConditionalSR 推理模式 (hard_mask=True) 前向传播成功。输出键: {list(output_eval_hard.keys())}")
        assert "sr_image" in output_eval_hard and output_eval_hard["sr_image"] is not None
        assert output_eval_hard["sr_image"].shape == (1, 3, 64 * 4, 64 * 4), "ConditionalSR 推理 (hard) sr_image 输出形状不匹配"
        assert "mask_coarse" in output_eval_hard and output_eval_hard["mask_coarse"] is not None
        assert output_eval_hard["mask_coarse"].min() >= 0.0 and output_eval_hard["mask_coarse"].max() <= 1.0 # 硬掩码是0或1
        logger_test.info(f"  SR 图像形状: {output_eval_hard['sr_image'].shape}, 粗粒度掩码形状: {output_eval_hard['mask_coarse'].shape}")

        # 测试推理模式下的前向传播 (使用软掩码/概率图)
        logger_test.info("测试 ConditionalSR 推理模式 (hard_mask=False) 前向传播...")
        output_eval_soft = conditional_sr_model(dummy_input_conditional_sr, hard_mask_inference=False)
        logger_test.info(f"ConditionalSR 推理模式 (hard_mask=False) 前向传播成功。输出键: {list(output_eval_soft.keys())}")
        assert "sr_image" in output_eval_soft and output_eval_soft["sr_image"] is not None
        assert output_eval_soft["sr_image"].shape == (1, 3, 64 * 4, 64 * 4), "ConditionalSR 推理 (soft) sr_image 输出形状不匹配"
        assert "mask_coarse" in output_eval_soft and output_eval_soft["mask_coarse"] is not None
        assert output_eval_soft["mask_coarse"].min() >= 0.0 and output_eval_soft["mask_coarse"].max() <= 1.0 # 概率图
        logger_test.info(f"  SR 图像形状: {output_eval_soft['sr_image'].shape}, 粗粒度掩码形状: {output_eval_soft['mask_coarse'].shape}")


        logger_test.info("所有模型初始化和基本前向传播测试通过！")

    except AssertionError as e_assert:
        logger_test.error(f"断言失败: {e_assert}", exc_info=True)
    except Exception as e_fwd:
        logger_test.error(f"ConditionalSR 前向传播时发生错误: {e_fwd}", exc_info=True)


if __name__ == "__main__":
    test_model_initialization_and_forward()