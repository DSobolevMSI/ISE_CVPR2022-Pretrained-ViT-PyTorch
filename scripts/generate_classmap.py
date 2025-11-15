import os

# 替换为你的 SO32 绝对路径
dataset_root = '/ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch/dataset/SO32/train'  # e.g., '/ldap_shared/home/s_ljy/dataset/SO32'

# 扫描子文件夹（类名，按字母排序赋 ID）
class_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
assert len(class_names) == 32, f"Found {len(class_names)} classes, expected 32!"

# 保存为 .txt（每行一个类名）
output_file = '/ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch/dataset/SO32_class_map.txt'  # 或 class_to_idx.txt
with open(output_file, 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"Class map saved to {output_file}")