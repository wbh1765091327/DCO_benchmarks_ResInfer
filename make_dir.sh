source set.sh
# mkdir ./DATA
# mkdir ./results
# mkdir ./results/recall@20
# mkdir ./results/recall@100
# mkdir ./results/recall@10
# mkdir ./results/recall@1
# rm -r cmake-build-debug
# mkdir cmake-build-debug
# cd cmake-build-debug
# cmake ..
# make clean
# make -j 40

# cd ..

# mkdir ./logger
# mkdir ./figure
  mkdir ./results/recall@10
  mkdir ./results/recall@10/ivf
  mkdir ./results/recall@10/hnsw
  mkdir ./results/recall@10/hnsw/simd
  mkdir ./results/recall@10/hnsw/nosimd
  mkdir ./results/recall@10/ivf/simd
  mkdir ./results/recall@10/ivf/nosimd

for dataset in "${datasets[@]}";
do
  echo $dataset
  # mkdir ./DATA/${dataset}
  mkdir ./DATA/${dataset}/linear
  mkdir ./results/recall@10/hnsw/simd/${dataset}
  mkdir ./results/recall@10/hnsw/nosimd/${dataset}
  mkdir ./results/recall@10/ivf/simd/${dataset}
  mkdir ./results/recall@10/ivf/nosimd/${dataset}

  mkdir ./logger/${dataset}
done