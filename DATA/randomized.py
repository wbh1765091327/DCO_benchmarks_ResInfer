import os
import numpy as np
import struct
from utils import fvecs_write, fvecs_read
import argparse
source = '/home/wbh/cppwork/Res-Infer/DATA/'

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

# datasets = ['nytimes-16-angular', 
#             'fashion-mnist-784-euclidean', 
#             'glove-50-angular', 
#             'glove-200-angular',
#             'sift-128-euclidean',
#             'msong-420', 
#             'contriever-768', 
#             'gist-960-euclidean', 
#             'deep-image-96-angular', 
#             'instructorxl-arxiv-768', 
#             'openai-1536-angular']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    np.random.seed(0)

    # path
    path = os.path.join(source, f'{dataset}')
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = fvecs_read(data_path)
    D = X.shape[1]

    # generate random orthogonal matrix, store it and apply it
    print(f"Randomizing {dataset} of dimensionality {D}.")
    P = Orthogonal(D)
    XP = np.dot(X, P)

    projection_path = f'./DATA/{dataset}/O.fvecs'
    transformed_path = f'./DATA/{dataset}/O{dataset}_base.fvecs'

    fvecs_write(projection_path, P)
    fvecs_write(transformed_path, XP)