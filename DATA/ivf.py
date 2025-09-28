import numpy as np
import faiss
import struct
import os
from utils import fvecs_write, fvecs_read
import argparse
from numpy.random import default_rng
source = '/home/wbh/cppwork/Res-Infer/DATA'
pre_source = './DATA'
# the number of clusters
K = 4096

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-m', '--method', help='approximate method', default='pca')
    parser.add_argument('-k', '--K', help='number of clusters', type=int, default=4096)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    method = args['method']
    K = int(args['K'])
    source = os.getenv('store_path')
    print(source)
    print(f"Clustering - {dataset}")
    if method == "naive":
        path = os.path.join(source, f'{dataset}')
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        X = fvecs_read(data_path)
        D = X.shape[1]
        num_embeddings = X.shape[0]
        training_points = K * 50  # Our collections do not need that many training points
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        if training_points < num_embeddings:
            rng = default_rng()
            training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
            training_sample_idxs.sort()
            print('Training with', training_points)
            index.train(X[training_sample_idxs])
        else:
            print('Training with all points')
            index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
    elif method == 'O':
        data_path = f'./DATA/{dataset}/O{dataset}_base.fvecs'
        centroids_path = f"./DATA/{dataset}/O{dataset}_centroid_{K}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]
        num_embeddings = X.shape[0]
        training_points = K * 50  # Our collections do not need that many training points
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        if training_points < num_embeddings:
            rng = default_rng()
            training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
            training_sample_idxs.sort()
            print('Training with', training_points)
            index.train(X[training_sample_idxs])
        else:
            print('Training with all points')
            index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
    elif method == 'pca':
        data_path = f'./DATA/{dataset}/{dataset}_base_{method}.fvecs'
        centroids_path = f"./DATA/{dataset}/{dataset}_centroid_{method}_{K}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]
        num_embeddings = X.shape[0]
        training_points = K * 50  # Our collections do not need that many training points
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        if training_points < num_embeddings:
            rng = default_rng()
            training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
            training_sample_idxs.sort()
            print('Training with', training_points)
            index.train(X[training_sample_idxs])
        else:
            print('Training with all points')
            index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
    elif method == 'opq':
        data_path = f'./DATA/{dataset}/{dataset}_base_{method}.fvecs'
        centroids_path = f"./DATA/{dataset}/{dataset}_centroid_{method}_{K}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]  
        num_embeddings = X.shape[0]
        training_points = K * 50  # Our collections do not need that many training points
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        if training_points < num_embeddings:
            rng = default_rng()
            training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
            training_sample_idxs.sort()
            print('Training with', training_points)
            index.train(X[training_sample_idxs])
        else:
            print('Training with all points')
            index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)