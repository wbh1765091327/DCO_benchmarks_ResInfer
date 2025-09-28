import os
import numpy as np
from utils import ivecs_read, ivecs_write

def process_groundtruth(source_dir, datasets, k=100, new_k=10):
    """
    处理所有数据集的groundtruth文件，截取前new_k个作为新的groundtruth
    
    参数:
    source_dir: 数据根目录
    datasets: 数据集名称列表
    k: 原始groundtruth的k值
    new_k: 新的groundtruth的k值
    """
    for dataset in datasets:
        print(f"\n处理数据集: {dataset}")
        
        # 构建文件路径
        path = os.path.join(source_dir, dataset)
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
        new_gt_path = os.path.join(path, f'{dataset}_groundtruth{new_k}.ivecs')
        
        # 读取原始groundtruth
        print(f"读取 {gt_path}")
        gt = ivecs_read(gt_path)
        
        # 截取前new_k个
        new_gt = gt[:, :new_k]
        
        # 保存新的groundtruth
        print(f"保存到 {new_gt_path}")
        ivecs_write(new_gt_path, new_gt)
        
        print(f"数据集 {dataset} 处理完成")

if __name__ == "__main__":
    # 数据根目录
    source_dir = '/home/wbh/cppwork/SuCo/data'
    
    # 数据集列表
    datasets = [
        'deep-image-96-angular', 
        'instructorxl-arxiv-768', 
        'openai-1536-angular'
    ]
    
    # 处理所有数据集
    process_groundtruth(source_dir, datasets, k=100, new_k=10)