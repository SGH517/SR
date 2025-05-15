import torch
import argparse
from models.conditional_sr import ConditionalSR
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker

def parse_args():
    parser = argparse.ArgumentParser(description="Test model initialization and forward pass")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing if available")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for faster GPU transfer")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    return parser.parse_args()

def test_model_initialization():
    args = parse_args()
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    if args.use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available, falling back to CPU")
    print(f"Using device: {device}")

    # 测试SRFast初始化
    sr_fast = SRFast(scale_factor=4).to(device)
    print("SRFast model initialized successfully")

    # 测试SRQuality初始化
    sr_quality = SRQuality(scale_factor=4).to(device)
    print("SRQuality model initialized successfully")

    # 测试Masker初始化
    masker = Masker(in_channels=3, base_channels=32, num_blocks=4).to(device)
    print("Masker model initialized successfully")

    # 测试ConditionalSR初始化
    model = ConditionalSR(
        sr_fast=sr_fast,
        sr_quality=sr_quality,
        masker=masker,
        detector_weights="",
        sr_fast_weights="",
        sr_quality_weights="",
        masker_weights=None,
        device=device
    ).to(device)
    print("ConditionalSR model initialized successfully")

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    try:
        output = model(dummy_input)
        print("Model forward pass successful")
        print("Output keys:", output.keys())
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    test_model_initialization()
