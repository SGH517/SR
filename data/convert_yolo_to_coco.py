import os
import json
from PIL import Image

def convert_yolo_to_coco(image_dir, label_dir, output_file):
    """
    将YOLO格式标注转换为COCO格式的annotations.json
    
    参数:
        image_dir: 图像目录路径
        label_dir: YOLO标签目录路径
        output_file: 输出JSON文件路径
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "object"}]  # 假设只有一类
    }
    
    annotation_id = 1
    
    # 遍历图像目录
    for image_idx, image_name in enumerate(os.listdir(image_dir)):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            width, height = img.size
        
        # 添加图像信息
        image_info = {
            "id": image_idx + 1,
            "file_name": image_name,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)
        
        # 处理标注文件
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id, x_center, y_center, w, h = map(float, parts)
                
                # 转换为绝对坐标
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                
                # 计算边界框坐标 (x_min, y_min, width, height)
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                
                # 添加标注信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_idx + 1,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
    
    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    # 示例用法
    image_dir = "D:\\论文相关\\Data partitioning\\train\\images"  # 需要确认图像目录路径
    label_dir = "D:\\论文相关\\Data partitioning\\train\\labels"
    output_file = "D:\\论文相关\\Data partitioning\\train\\annotations.json"
    
    convert_yolo_to_coco(image_dir, label_dir, output_file)
    print(f"转换完成，结果已保存到 {output_file}")
