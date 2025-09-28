import numpy as np
from sklearn.decomposition import PCA
from utils import fvecs_read, fvecs_write, ivecs_read, ivecs_write
import os
import argparse
import struct

source = '/home/wbh/cppwork/Res-Infer/DATA/'

datasets = ['nytimes-16-angular', 
            'fashion-mnist-784-euclidean', 
            'glove-50-angular', 
            'glove-200-angular',
            'sift-128-euclidean',
            'msong-420', 
            'contriever-768', 
            'gist-960-euclidean', 
            'deep-image-96-angular', 
            'instructorxl-arxiv-768', 
            'openai-1536-angular']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    print(f"PCA - {dataset}")
    # path
    path = os.path.join(source, f'{dataset}')
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    ground_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')
    # read data vectors
    base = fvecs_read(base_path)
    N, D = base.shape
    pca_dim = D
    # projection
    mean = np.mean(base, axis=0)
    base -= mean
    pca = PCA(n_components=pca_dim)
    if N < 10000000:
        pca.fit(base)
    else:
        pca.fit(base[:10000000])
    # save the transpose matrix
    projection_matrix = pca.components_.T
    base = np.dot(base, projection_matrix)
    print(f"PCA - finished")
    save_base_path = f'./DATA/{dataset}/{dataset}_base_pca.fvecs'
    matrix_save_path = f'./DATA/{dataset}/{dataset}_pca_matrix.fvecs'

    variance = np.var(base, axis=0)
    save_matrix = np.vstack((mean, mean, variance, projection_matrix))
    fvecs_write(matrix_save_path, save_matrix)
    fvecs_write(save_base_path, base)

    for K in [10]:
        matrix_save_path = f'./DATA/{dataset}/{dataset}_pca_matrix_{K}.fvecs'
        ground = ivecs_read(ground_path)
        ground = ground[:int(1e4), :K]
        ground = ground.flatten()
        X_sample = base[ground]
        sample_mean = np.mean(X_sample, axis=0)
        X_sample -= sample_mean
        variance = np.var(X_sample, axis=0)
        save_matrix = np.vstack((mean, sample_mean, variance, projection_matrix))
        fvecs_write(matrix_save_path, save_matrix)
