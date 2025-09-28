import numpy as np
import faiss
import struct
import os
from utils import fvecs_read, fvecs_write
import argparse
from numpy.random import default_rng

source = '/home/wbh/cppwork/Res-Infer/DATA'
M = 64
nbits = 8

def save_centroid(filename, data):
    print(f"Writing centroid file - {filename}")
    M, k, d = data.shape
    with open(filename, 'wb') as fp:
        item = struct.pack('I', M)
        fp.write(item)
        item = struct.pack('I', k)
        fp.write(item)
        item = struct.pack('I', d)
        fp.write(item)
        for x in data:
            for y in x:
                for z in y:
                    a = struct.pack('f', z)
                    fp.write(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-b', '--bits', help='cluster bits', default=8)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    nbits = int(args['bits'])
    print(f"OPQ transform - {dataset}")
    path = os.path.join(source, f'{dataset}')
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    X_base = fvecs_read(data_path)
    d = X_base.shape[1]
    M = 64
    if dataset == "gist-960-euclidean" or dataset == "openai-1536-angular" or dataset == "instructorxl-arxiv-768_100k" or dataset == "contriever-768" or dataset == "instructorxl-arxiv-768_1k" or dataset == "instructorxl-arxiv-768_10k" or dataset == "instructorxl-arxiv-768_1000k":
        M = int(d / 8)
    elif dataset == "glove-25-angular_100k" or dataset == "glove-50-angular_100k" or dataset == "glove-100-angular_100k" or dataset == "glove-200-angular_10k" or dataset == "glove-200-angular_1k": 
        M = int(d / 5)
    else:
        M = int(d / 4)

    print(f"Dataset shape: {X_base.shape}")
    ntrain = min(X_base.shape[0], 256000)
    print(f"Using {ntrain} vectors for training OPQ and PQ.")
    rng = default_rng()
    train_indices = rng.choice(X_base.shape[0], size=ntrain, replace=False)
    train_indices.sort()
    X_train = X_base[train_indices]

    d2 = ((d + M - 1) // M) * M
    opq = faiss.OPQMatrix(d, M, d2)
    opq.verbose = False
    print("Training OPQ matrix on the sample...")
    opq.train(X_train)
    Matrix_A = faiss.vector_float_to_array(opq.A)
    Matrix_A = Matrix_A.reshape(d2, d2)
    # save the transpose matrix
    fvecs_write(f'./DATA/{dataset}/{dataset}_opq_matrix.fvecs', Matrix_A.T)
    
    print("Applying OPQ transformation to the full dataset...")
    X_base = opq.apply(X_base)
    
    # For PQ training, we use the transformed vectors from our training sample
    X_train_pq = X_base[train_indices]
    
    pq = faiss.ProductQuantizer(d2, M, nbits)
    pq.verbose = False
    print("Training PQ codebooks on the transformed sample...")
    pq.train(X_train_pq)
    centroids = faiss.vector_float_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
    save_centroid(f'./DATA/{dataset}/{dataset}_codebook.centroid', centroids)
    fvecs_write(f'./DATA/{dataset}/{dataset}_base_opq.fvecs', X_base)
