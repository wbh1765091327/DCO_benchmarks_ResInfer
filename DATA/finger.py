import hnswlib
import numpy as np
import struct
import os
import time
import logging
from datetime import datetime
from sklearn.decomposition import PCA

def ed(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def generate_matrix(lowdim, D):
    return np.random.normal(size=(lowdim, D))

def get_compact_sign_matrix(sgn_m, dt):
    return np.packbits(sgn_m, axis=1, bitorder='little').view(dt)

def read_hnsw_index(filepath, D):
    '''
    Read hnsw index from binary file
    '''
    index = hnswlib.Index(space='l2', dim=D)
    index.load_index(filepath)
    return index

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def to_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

# return internal ids
def get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element):
    start = int(internal_id * size_data_per_element / 4)
    cnt = data_level_0[start]
    neighbors = []
    for i in range(cnt):
        neighbors.append(data_level_0[start + i + 1])
    return neighbors

# return internal ids
def get_neighbors_with_external_label(data_level_0, external_label, size_data_per_element, label2id):
    internal_id = label2id[external_label]
    return get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element)

def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'finger_preprocessing_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)

source = '/home/wbh/cppwork/Res-Infer/DATA'
# datasets = ['glove-200-angular', 'sift-128-euclidean', 'msong-420', 'contriever-768', 
#            'gist-960-euclidean', 'deep-image-96-angular', 'instructorxl-arxiv-768', 'openai-1536-angular']
datasets = ['glove-25-angular_100k']

ef = 500
M = 16
lsh_dim = 16

if __name__ == '__main__':
    # 设置日志
    logger = setup_logging()
    
    # 记录总体开始时间
    total_start_time = time.time()
    logger.info("=" * 80)
    logger.info("开始Finger预处理任务")
    logger.info(f"数据集列表: {datasets}")
    logger.info(f"参数设置: ef={ef}, M={M}, lsh_dim={lsh_dim}")
    logger.info("=" * 80)
    
    # 统计信息
    total_datasets = len(datasets)
    successful_datasets = 0
    failed_datasets = []
    
    for dataset_idx, dataset in enumerate(datasets, 1):
        dataset_start_time = time.time()
        logger.info(f"[{dataset_idx}/{total_datasets}] 开始处理数据集: {dataset}")
        
        try:
            # 记录各阶段时间
            stage_times = {}
            
            # 阶段1: 读取数据
            stage_start = time.time()
            path = os.path.join(source, f'{dataset}')
            index_path = os.path.join(path, f'{dataset}_ef{ef}_M{M}.index')
            data_path = os.path.join(path, f'{dataset}_base.fvecs')
            
            logger.info(f"  读取数据文件: {data_path}")
            X = read_fvecs(data_path)
            logger.info(f"  数据形状: {X.shape}")
            
            logger.info(f"  读取索引文件: {index_path}")
            index = read_hnsw_index(index_path, X.shape[1])
            stage_times['读取数据'] = time.time() - stage_start
            
            # 阶段2: 提取索引信息
            stage_start = time.time()
            ann_data = index.get_ann_data()
            data_level_0 = ann_data['data_level0']
            size_data_per_element = ann_data['size_data_per_element']
            offset_data = ann_data['offset_data']
            internal_ids = ann_data['label_lookup_internal']
            external_ids = ann_data['label_lookup_external']
            
            id2label = {}
            label2id = {}
            data_level_0 = data_level_0.view("int32")
            for i in range(len(internal_ids)):
                id2label[internal_ids[i]] = external_ids[i]
                label2id[external_ids[i]] = internal_ids[i]
            stage_times['提取索引信息'] = time.time() - stage_start
            
            # 阶段3: 获取邻居信息
            stage_start = time.time()
            logger.info("  获取每个节点的邻居信息")
            data_size = X.shape[0]
            total_cnt = 0
            startIdx = np.zeros(data_size, dtype=np.int32)
            
            for i in range(data_size):
                startIdx[i] = total_cnt
                neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
                total_cnt += len(neighbors)
            stage_times['获取邻居信息'] = time.time() - stage_start
            
            # 阶段4: 构建d_res向量
            stage_start = time.time()
            logger.info("  构建d_res向量")
            d_res_vecs = []
            sample_size = min(100000, data_size)
            sample_indices = np.random.choice(data_size, sample_size, replace=False)
            
            for i in sample_indices:
                neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
                nei_labels = [id2label[nei] for nei in neighbors]
                nei_vecs = X[nei_labels]
                c_2 = X[i] @ X[i]
                if c_2 != 0:
                    nei_proj_vecs = np.outer((nei_vecs @ X[i]) / c_2, X[i])
                    nei_res_vecs = nei_vecs - nei_proj_vecs
                    d_res_vecs += list(nei_res_vecs)
            stage_times['构建d_res向量'] = time.time() - stage_start
            
            # 阶段5: PCA计算
            stage_start = time.time()
            logger.info(f"  计算d_res的PCA (维度: {lsh_dim})")
            pca = PCA(n_components=lsh_dim)
            pca.fit(d_res_vecs)
            P = pca.components_.T
            stage_times['PCA计算'] = time.time() - stage_start
            
            # 阶段6: 主预处理
            stage_start = time.time()
            logger.info("  开始主预处理")
            c_2s = np.zeros(data_size, dtype=np.float32)
            node_info_float = np.zeros((data_size, lsh_dim), dtype=np.float32)
            edge_info_float = np.zeros((total_cnt, 2), dtype=np.float32)
            edge_info_uint = np.zeros((total_cnt, lsh_dim), dtype=np.uint32)
            
            cur_idx = 0
            for cur_label in range(data_size):
                if cur_label % 1000000 == 0:
                    logger.info(f"    处理进度: {cur_label}/{data_size}")
                    
                assert(cur_idx == startIdx[cur_label])
                cur_vec = X[cur_label]
                cur_c_2 = cur_vec @ cur_vec
                c_2s[cur_label] = cur_c_2
                neighbors = get_neighbors_with_external_label(data_level_0, cur_label, size_data_per_element, label2id)
                num_nei = len(neighbors)
                
                c_P = cur_vec @ P
                node_info_float[cur_label, :] = c_P
                
                if cur_c_2 == 0:
                    edge_info_float[cur_idx: cur_idx+num_nei, :] = np.zeros((num_nei, 2))
                else:
                    nei_labels = [id2label[nei] for nei in neighbors]
                    nei_vecs = X[nei_labels]
                    
                    bs = nei_vecs @ cur_vec / cur_c_2
                    d_2s = np.sum(nei_vecs * nei_vecs, axis=1, dtype=np.float32)
                    d_proj_2s = bs * bs * cur_c_2
                    d_res_2s = d_2s - d_proj_2s
                    if np.any(d_res_2s < 0):
                        d_res_2s[d_res_2s < 0] = 0
                    d_ress = np.sqrt(d_res_2s)
                    
                    edge_info_float[cur_idx: cur_idx+num_nei, :] = np.array([bs, d_ress]).T
                    
                    d_res_vecs_dot_P = nei_vecs @ P - np.outer(bs, cur_vec) @ P
                    edge_info_uint[cur_idx: cur_idx+num_nei, :] = (np.sign(d_res_vecs_dot_P) > 0)
                
                cur_idx += num_nei
            stage_times['主预处理'] = time.time() - stage_start
            
            # 阶段7: 保存文件
            stage_start = time.time()
            logger.info("  保存预处理文件")
            finger_path = os.path.join(source ,  f'{dataset}')
            projection_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_LSH.fvecs')
            b_dres_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_b_dres.fvecs')
            sgn_dres_P_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_sgn_dres_P.ivecs')
            c_2_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_c_2.fvecs')
            c_P_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_c_P.fvecs')
            start_idx_path = os.path.join(finger_path, f'FINGER{lsh_dim}_{dataset}M{M}ef{ef}_start_idx.ivecs')
            
            to_fvecs(projection_path, P)
            to_fvecs(b_dres_path, edge_info_float)
            to_ivecs(sgn_dres_P_path, edge_info_uint)
            to_fvecs(c_2_path, c_2s.astype(np.float32).reshape(data_size,1))
            to_fvecs(c_P_path, node_info_float)
            to_ivecs(start_idx_path, startIdx.reshape(data_size,1))
            stage_times['保存文件'] = time.time() - stage_start
            
            # 计算总时间
            dataset_total_time = time.time() - dataset_start_time
            successful_datasets += 1
            
            # 记录详细时间信息
            logger.info(f"  ✓ 数据集 {dataset} 处理完成")
            logger.info(f"    总耗时: {dataset_total_time:.2f}秒")
            logger.info(f"    各阶段耗时:")
            for stage, stage_time in stage_times.items():
                percentage = (stage_time / dataset_total_time) * 100
                logger.info(f"      {stage}: {stage_time:.2f}秒 ({percentage:.1f}%)")
            logger.info(f"    数据统计: 向量数={data_size}, 总边数={total_cnt}")
            logger.info("-" * 60)
            
        except Exception as e:
            dataset_total_time = time.time() - dataset_start_time
            failed_datasets.append(dataset)
            logger.error(f"  ✗ 数据集 {dataset} 处理失败")
            logger.error(f"    错误信息: {str(e)}")
            logger.error(f"    耗时: {dataset_total_time:.2f}秒")
            logger.error("-" * 60)
    
    # 记录总体统计信息
    total_time = time.time() - total_start_time
    logger.info("=" * 80)
    logger.info("Finger预处理任务完成")
    logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    logger.info(f"成功处理: {successful_datasets}/{total_datasets} 个数据集")
    if failed_datasets:
        logger.info(f"失败数据集: {failed_datasets}")
    logger.info("=" * 80)