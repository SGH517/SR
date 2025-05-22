# data/detection_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO # 请确保已安装: pip install pycocotools
import torchvision.transforms.functional as F # For F.to_tensor

class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, return_image_id=False):
        """
        Args:
            image_dir (str): 包含图像的目录。
                             例如在 stage3_finetune_joint.py 中, 这会是 os.path.join(config['dataset']['image_dir'], "LR")
            annotation_file (str): COCO 格式标注文件的路径。
            transform (callable, optional): 应用于图像的转换。
            return_image_id (bool): 如果为 True, target 字典将包含 'image_id'。
                                   (在 stage3_finetune_joint.py 中设置为 True)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.return_image_id = return_image_id # stage3_finetune_joint.py 中使用了此参数

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # (可选) 过滤掉没有标注的图像ID (如果您的任务需要)
        # valid_ids = []
        # for img_id in self.ids:
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id)
        #     if len(ann_ids) > 0:
        #         valid_ids.append(img_id)
        # self.ids = valid_ids
        # print(f"Initialized DetectionDataset with {len(self.ids)} images that have annotations.")


    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)  # 此图像的标注列表

        img_info = coco.loadImgs(img_id)[0]
        coco_file_name = img_info['file_name']
        path = os.path.join(self.image_dir, coco_file_name) 
        path = os.path.normpath(path)

        try:
            img = Image.open(path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: 图像文件未找到: {path} (img_id: {img_id}, coco_file_name: {coco_file_name}, dataset_image_dir: {self.image_dir})")
            # 返回 None, 以便 collate_fn 可以过滤掉它
            return None, None 
        except Exception as e:
            print(f"ERROR: 加载图像时出错 {path}: {e}")
            return None, None

        # 准备 target 字典
        target = {}
        boxes = [ann['bbox'] for ann in anns]  # COCO format: [x_min, y_min, width, height]
                                               # 这些坐标应已由 prepare_detection_data.py 缩放到 LR 图像尺寸

        # 转换为张量并处理空 boxes 的情况
        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            # 将 [x_min, y_min, width, height] 转换为 [x_min, y_min, x_max, y_max]
            boxes_tensor[:, 2:] += boxes_tensor[:, :2]
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        
        target['boxes'] = boxes_tensor

        labels = [ann['category_id'] for ann in anns]
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        if self.return_image_id: # 根据 stage3_finetune_joint.py 中的使用情况添加
            target['image_id'] = torch.tensor([img_id])
        
        # (可选) 添加其他 COCO 标准字段，如果模型或损失函数需要
        # areas = [ann.get('area', 0.0) for ann in anns] # 使用 .get 避免 KeyError
        # target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        
        # iscrowd = [ann.get('iscrowd', 0) for ann in anns]
        # target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # 原始图像尺寸 (LR 图像的尺寸)，这对于某些模型和评估可能有用
        # target['orig_size'] = torch.as_tensor([int(img_info['height']), int(img_info['width'])]) # H, W

        if self.transform:
            img = self.transform(img) # 通常包含 ToTensor()
        else:
            # 如果没有提供 transform，确保图像被转换为张量
            img = F.to_tensor(img) # from torchvision.transforms.functional import to_tensor

        # 确保 target 中的所有值都是张量
        # (上面已经对 boxes, labels, image_id 做了处理)

        return img, target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        """
        自定义的 collate_fn 用于处理目标检测的批次。
        它会过滤掉在 __getitem__ 中可能因图像加载失败而返回 (None, None) 的项。
        """
        # 过滤掉无效的样本 (图像加载失败等)
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        
        if not batch: # 如果整个批次都无效
            # 根据您的错误处理策略，可以返回 None 或引发错误
            # 或者返回空的批次 (但这需要下游代码能处理)
            print("WARNING: Collate function received an empty batch after filtering None items.")
            # 返回一个可接受的空批次结构，或者让调用者处理
            # 例如，如果下游期望 (images_tensor, targets_list):
            return torch.empty(0), [] 

        images, targets = zip(*batch)
        
        # 将图像堆叠成一个批次张量
        try:
            images = torch.stack(images, 0)
        except RuntimeError as e:
            print(f"Error stacking images in collate_fn: {e}")
            print(f"Number of images: {len(images)}")
            for i, img_tensor in enumerate(images):
                print(f"Image {i} shape: {img_tensor.shape}")
            raise

        return images, list(targets) # targets 是一个字典列表