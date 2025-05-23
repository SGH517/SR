# data/detection_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError # 导入 UnidentifiedImageError
from pycocotools.coco import COCO # 请确保已安装: pip install pycocotools
import torchvision.transforms.functional as TF # 使用 TF 作为别名，避免与 F (torch.nn.functional) 混淆
from typing import Tuple, Dict, List, Optional, Any # 增加类型提示
import logging

# 为此模块设置一个 logger
logger_det_dataset = logging.getLogger(__name__)
if not logger_det_dataset.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 该模块定义了目标检测数据集类，用于加载和处理检测任务的数据。

class DetectionDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 annotation_file: str,
                 transform: Optional[callable] = None,
                 return_image_id: bool = False):
        """
        Args:
            image_dir (str): 包含图像的目录。
                             对于由 prepare_detection_data.py 处理后的数据，
                             这通常是 output_dir_base (例如 "dataset/date_prepared")，
                             因为标注文件中的 file_name 会是 "LR/image.jpg"。
            annotation_file (str): COCO 格式标注文件的路径 (例如 "dataset/date_prepared/annotations_lr.json")。
            transform (callable, optional): 应用于图像的转换。
            return_image_id (bool): 如果为 True, target 字典将包含 'image_id'。
                                   (在 stage3_finetune_joint.py 和 evaluate.py 中通常设为 True)
        """
        self.image_dir = image_dir # 这是图像文件相对于标注文件中 file_name 字段的根目录
        self.transform = transform
        self.return_image_id = return_image_id

        if not os.path.isfile(annotation_file): # 检查文件是否存在且是文件
            logger_det_dataset.error(f"标注文件未找到或不是一个文件: {annotation_file}")
            raise FileNotFoundError(f"标注文件未找到或不是一个文件: {annotation_file}")
        # image_dir 的存在性将在 __getitem__ 中通过 os.path.join 构建完整路径时隐式检查
        # 如果需要，也可以在这里添加 os.path.isdir(image_dir) 的检查

        try:
            self.coco = COCO(annotation_file)
        except Exception as e:
            logger_det_dataset.error(f"加载 COCO 标注文件 {annotation_file} 失败: {e}", exc_info=True)
            raise
        
        self.ids = list(sorted(self.coco.imgs.keys())) # 获取所有图像ID并排序

        if not self.ids:
            logger_det_dataset.warning(f"从标注文件 {annotation_file} 中未加载到任何图像ID。数据集将为空。")

        # (可选) 过滤掉没有标注的图像ID
        # 如果您的任务严格要求每张图像都有标注，可以取消注释以下代码块
        # initial_num_ids = len(self.ids)
        # valid_ids = []
        # for img_id in self.ids:
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id)
        #     if len(ann_ids) > 0:
        #         valid_ids.append(img_id)
        # self.ids = valid_ids
        # if initial_num_ids > len(self.ids):
        #     logger_det_dataset.info(
        #         f"已过滤掉 {initial_num_ids - len(self.ids)} 张没有标注的图像。"
        #         f"剩余图像数量: {len(self.ids)}"
        #     )
        logger_det_dataset.info(f"DetectionDataset 初始化完成，包含 {len(self.ids)} 张图像 "
                                f"(基于标注文件: {annotation_file})。")


    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        获取指定索引处的图像和其对应的标注。

        返回:
            Tuple[torch.Tensor, Dict[str, Any]]: 图像张量和包含标注信息的字典。
                                                如果发生错误，则返回 (None, None)，由 collate_fn 处理。
        """
        if not (0 <= index < len(self.ids)):
            logger_det_dataset.error(f"索引 {index} 超出范围 (0-{len(self.ids)-1})。")
            return None, None # 由 collate_fn 处理

        coco_instance = self.coco
        img_id = self.ids[index]

        # 获取此图像的所有标注ID和标注内容
        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        anns_for_image: List[Dict[str, Any]] = coco_instance.loadAnns(ann_ids)

        img_info: Dict[str, Any] = coco_instance.loadImgs(img_id)[0]
        # file_name 在 COCO json 中是相对于图像根目录的路径
        # 例如 "LR/image_name.jpg"
        relative_img_path = img_info['file_name']
        # image_dir 是这个根目录，例如 "output_dir_base_from_prepare_script"
        full_img_path = os.path.join(self.image_dir, relative_img_path)
        full_img_path = os.path.normpath(full_img_path) # 规范化路径

        try:
            img_pil = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            logger_det_dataset.warning(f"图像文件未找到: {full_img_path} (图像ID: {img_id}, "
                                    f"COCO文件名: '{relative_img_path}', 数据集根目录: '{self.image_dir}')")
            return None, None
        except UnidentifiedImageError:
            logger_det_dataset.warning(f"无法识别的图像文件 (可能已损坏): {full_img_path} (图像ID: {img_id})")
            return None, None
        except Exception as e:
            logger_det_dataset.error(f"加载图像 {full_img_path} (图像ID: {img_id}) 时发生错误: {e}", exc_info=True)
            return None, None

        # --- 准备 target 字典 ---
        target: Dict[str, Any] = {}
        
        # 从标注中提取边界框 (COCO格式: [x_min, y_min, width, height])
        # 这些坐标应该已经是相对于LR图像尺寸的了（由 prepare_detection_data.py 处理过）
        boxes_coco_fmt = [ann['bbox'] for ann in anns_for_image if 'bbox' in ann]

        if boxes_coco_fmt:
            boxes_tensor_coco = torch.as_tensor(boxes_coco_fmt, dtype=torch.float32).reshape(-1, 4)
            # 将 COCO 的 [x_min, y_min, width, height] 转换为 [x_min, y_min, x_max, y_max]
            # 这是许多检测模型（如 torchvision 的 Faster R-CNN）期望的格式
            boxes_tensor_xyxy = boxes_tensor_coco.clone()
            boxes_tensor_xyxy[:, 2] += boxes_tensor_coco[:, 0] # x_max = x_min + width
            boxes_tensor_xyxy[:, 3] += boxes_tensor_coco[:, 1] # y_max = y_min + height
        else:
            boxes_tensor_xyxy = torch.empty((0, 4), dtype=torch.float32)

        target['boxes'] = boxes_tensor_xyxy

        # 提取类别标签
        labels = [ann['category_id'] for ann in anns_for_image if 'category_id' in ann]
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        # 如果需要，添加 image_id (对于 COCO 评估是必需的)
        if self.return_image_id:
            target['image_id'] = torch.tensor([img_id], dtype=torch.int64) # 确保是张量和正确类型

        # (可选) 添加其他 COCO 标准字段，如果模型或损失函数需要
        # areas = [ann.get('area', 0.0) for ann in anns_for_image] # 使用 .get 避免 KeyError
        # target['area'] = torch.as_tensor(areas, dtype=torch.float32)

        # iscrowd = [ann.get('iscrowd', 0) for ann in anns_for_image]
        # target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # 原始图像尺寸 (LR 图像的尺寸)，这对于某些模型和评估可能有用
        # target['orig_size'] = torch.as_tensor([int(img_info['height']), int(img_info['width'])], dtype=torch.int64) # H, W

        # 应用图像转换
        img_tensor: torch.Tensor
        if self.transform:
            try:
                img_tensor = self.transform(img_pil)
            except Exception as e_transform:
                logger_det_dataset.warning(f"对图像 {full_img_path} (图像ID: {img_id}) 应用转换时出错: {e_transform}")
                return None, None
        else:
            # 如果没有提供 transform，确保图像被转换为张量
            img_tensor = TF.to_tensor(img_pil) # from torchvision.transforms.functional

        # 再次确认 target 中的所有值都是张量（上面已处理）

        return img_tensor, target

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def collate_fn(batch: List[Optional[Tuple[torch.Tensor, Dict[str, Any]]]]
                   ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        """
        自定义的 collate_fn 用于处理目标检测的批次。
        它会过滤掉在 __getitem__ 中可能因图像加载失败而返回 (None, None) 的项。
        """
        # 过滤掉无效的样本 (例如，图像加载失败导致 __getitem__ 返回 (None, None))
        valid_batch_items = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]

        if not valid_batch_items: # 如果整个批次都无效
            logger_det_dataset.warning("Collate 函数接收到空批次 (所有样本均无效或加载失败)。")
            # 返回一个可接受的空批次结构，或者让调用者处理
            # 例如，如果下游期望 (images_tensor, targets_list):
            return torch.empty(0, 3, 1, 1), [] # 返回一个形状合理的空图像张量和空目标列表

        images, targets = zip(*valid_batch_items)

        # 将图像堆叠成一个批次张量
        try:
            images_stacked = torch.stack(images, 0)
        except RuntimeError as e_stack: # 通常发生在图像尺寸不一致时
            logger_det_dataset.error(f"在 collate_fn 中堆叠图像时发生错误 (可能尺寸不一致): {e_stack}", exc_info=True)
            # 可以尝试记录每张图像的形状以帮助调试
            # for i, img_tensor_item in enumerate(images):
            #     logger_det_dataset.debug(f"  图像 {i} 形状: {img_tensor_item.shape}")
            # 返回一个可接受的空批次结构或重新引发错误
            return torch.empty(0, 3, 1, 1), [] # 或者 raise e_stack
        except Exception as e_gen_stack:
            logger_det_dataset.error(f"在 collate_fn 中堆叠图像时发生未知错误: {e_gen_stack}", exc_info=True)
            return torch.empty(0, 3, 1, 1), []


        return images_stacked, list(targets) # targets 是一个字典列表