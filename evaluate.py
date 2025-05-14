import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import setup_logger
from utils.losses import calculate_joint_loss
from utils.optimizer_utils import get_optimizer_with_differential_lr
from utils.evaluation_utils import run_coco_evaluation
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from torch.utils.tensorboard import SummaryWriter
import json  # For saving eval results
from pycocotools.coco import COCO  # For eval
from pycocotools.cocoeval import COCOeval  # For eval
from torchvision import transforms  # For eval dataset transform


def load_model_for_eval(args, device):
    print(f"Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return None
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # --- Try to load config from checkpoint ---
        config = checkpoint.get('config', None)
        model = None

        if config:
            print("Loading model configuration from checkpoint.")
            try:
                sr_fast_args = config.get('model', {}).get('sr_fast', {})
                sr_quality_args = config.get('model', {}).get('sr_quality', {})
                masker_args = config.get('model', {}).get('masker', {})

                sr_fast = SRFast(**sr_fast_args).to(device)
                sr_quality = SRQuality(**sr_quality_args).to(device)
                masker = Masker(**masker_args).to(device)

                model = ConditionalSR(
                    sr_fast=sr_fast,
                    sr_quality=sr_quality,
                    masker=masker,
                    detector_weights="",  # Detector should be loaded within ConditionalSR if path in config
                    sr_fast_weights="",   # Weights loaded via state_dict
                    sr_quality_weights="",  # Weights loaded via state_dict
                    masker_weights=None,  # Weights loaded via state_dict
                    device=device,
                    config=config  # Pass the loaded config
                ).to(device)
                print("Model instantiated using config from checkpoint.")
            except Exception as e:
                print(f"Error instantiating model using config from checkpoint: {e}. Falling back.")
                model = None

        if model is None:
            print("Warning: Could not load config from checkpoint or instantiation failed. Instantiating model components with default parameters.")
            sr_fast = SRFast().to(device)
            sr_quality = SRQuality().to(device)
            masker = Masker().to(device)
            mock_config = {'model': {'masker': {'threshold': 0.5}}, 'train': {}}
            model = ConditionalSR(sr_fast, sr_quality, masker, "", "", "", None, device, mock_config).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state_dict from {args.checkpoint}")
        elif isinstance(checkpoint, dict):
            try:
                model.load_state_dict(checkpoint)
                print(f"Loaded model state_dict directly from checkpoint file {args.checkpoint}")
            except RuntimeError as e:
                print(f"Failed to load state_dict directly: {e}. Checkpoint format might be incompatible.")
                return None
        else:
            print("Error: Checkpoint format not recognized or missing 'model_state_dict'.")
            return None

        model.eval()
        return model

    except Exception as e:
        print(f"Error loading checkpoint from {args.checkpoint}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConditionalSR model for Detection Performance")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory containing low-resolution images for evaluation.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the ground truth annotation file (JSON format) for evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the ConditionalSR checkpoint (.pth file).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--hard_mask", action='store_true', help="Use hard mask during inference.")
    parser.add_argument("--output_dir", type=str, default="evaluation_output", help="Directory to save detection results JSON.")
    args = parser.parse_args()

    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir, "evaluation.log")
    logger.info(f"Evaluation Arguments: {args}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model_for_eval(args, device)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return

    try:
        transform = transforms.ToTensor()
        dataset = DetectionDataset(image_dir=args.lr_dir, annotation_file=args.annotation_file, transform=transform, return_image_id=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=getattr(dataset, 'collate_fn', None))
        logger.info(f"Loaded dataset from {args.lr_dir} with {len(dataset)} images.")
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        return
    except AttributeError:
        logger.warning("Dataset does not support 'return_image_id'. Image IDs might be incorrect for COCO eval.")
        dataset = DetectionDataset(image_dir=args.lr_dir, annotation_file=args.annotation_file, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=getattr(dataset, 'collate_fn', None))
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset loading: {e}")
        return

    logger.info("Starting evaluation...")
    map_results, avg_sparsity = run_coco_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        step_or_epoch="final",
        logger=logger,
        use_hard_mask=args.hard_mask
    )

    logger.info("Evaluation finished.")
    logger.info(f"Average Sparsity (Quality Path Usage): {avg_sparsity:.4f}")
    if map_results:
        logger.info("Detection mAP Results:")
        logger.info(f"  mAP@0.50:95 = {map_results.get('map', 0.0):.4f}")
        logger.info(f"  mAP@0.50    = {map_results.get('map_50', 0.0):.4f}")
    else:
        logger.warning("mAP results are not available.")


if __name__ == "__main__":
    main()