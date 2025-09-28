from utils import fvecs_read
from utils import fvecs_write, ivecs_read, ivecs_write
import os
import numpy as np
import faiss
import time

source = '/home/wbh/cppwork/HNSW-Flash/data/'
datasets = [
            'nytimes-16-angular', 
            'glove-50-angular', 
            'glove-200-angular',
            'sift-128-euclidean',
            'msong-420', 
            'contriever-768', 
            'gist-960-euclidean', 
            'deep-image-96-angular', 
            'instructorxl-arxiv-768', 
            'openai-1536-angular']
# datasets = ['gist-960-euclidean', 
#             'deep-image-96-angular', 
#             'instructorxl-arxiv-768', 
#             'openai-1536-angular']


def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.verbose = True
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')


if __name__ == "__main__":
    total_start_time = time.time()
    
    for dataset in datasets:
        print(f'\n处理数据集: {dataset}')
        dataset_start_time = time.time()
        
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        
        print("读取数据...")
        base = fvecs_read(base_path)
        learn = fvecs_read(query_path)
        # learn_num = int(1e4)
        # learn_data = learn[:learn_num]
        
        print("计算ground truth...")
        gt = do_compute_gt(base, learn, topk=100)
        
        save_path = os.path.join(source, f'{dataset}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_ground_path = os.path.join(save_path, f'{dataset}_groundtruth.ivecs')
        ivecs_write(save_ground_path, gt)
        
        dataset_end_time = time.time()
        print(f"数据集 {dataset} 处理完成，耗时: {dataset_end_time - dataset_start_time:.2f} 秒")
    
    total_end_time = time.time()
    print(f"\n所有数据集处理完成，总耗时: {total_end_time - total_start_time:.2f} 秒")
