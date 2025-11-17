import os
import pandas as pd
from collections import defaultdict

dataset_root = '/ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch/dataset/SO32'
train_dir = os.path.join(dataset_root, 'train')
val_dir = os.path.join(dataset_root, 'val')

# 字典存储每个类的计数
train_counts = defaultdict(int)
val_counts = defaultdict(int)

# 统计 train
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        # 计数图像文件（假设扩展名为 .jpg, .png, .jpeg 等；可扩展）
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        count = sum(1 for file in os.listdir(class_path) 
                    if os.path.splitext(file.lower())[1] in image_extensions)
        train_counts[class_name] = count

# 统计 val
for class_name in os.listdir(val_dir):
    class_path = os.path.join(val_dir, class_name)
    if os.path.isdir(class_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        count = sum(1 for file in os.listdir(class_path) 
                    if os.path.splitext(file.lower())[1] in image_extensions)
        val_counts[class_name] = count

# 合并所有类（取并集）
all_classes = set(train_counts.keys()) | set(val_counts.keys())

# 创建 DataFrame
data = []
for class_name in sorted(all_classes):
    data.append({
        'class_name': class_name,
        'train_count': train_counts[class_name],
        'val_count': val_counts[class_name]
    })

df = pd.DataFrame(data)

# 添加总计行
total_row = pd.DataFrame({
    'class_name': ['Total'],
    'train_count': [df['train_count'].sum()],
    'val_count': [df['val_count'].sum()]
})
df = pd.concat([df, total_row], ignore_index=True)

# 保存到 CSV
output_csv = '/ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch/dataset/SO32_class_counts.csv'
df.to_csv(output_csv, index=False, encoding='utf-8')
print(df)  # 打印预览（最后一行是 Total）