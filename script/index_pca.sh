cd ..
source set.sh

efConstruction=500
M=16
# datasets=(
# #   "glove-25-angular_100k"
#   # "glove-50-angular_100k"
#   # "glove-100-angular_100k"
#   # "glove-200-angular_100k"
#   "glove-200-angular_10k"
# )、
datasets=(
    "instructorxl-arxiv-768_1k"
)
for data in "${datasets[@]}"; do
  echo "precompute - ${data}"
  log_file="./logger/${data}/PCA-Train-time.log"
  start_time=$(date +%s)
  python3 ./DATA/pcanew.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "PCA Train time: ${duration}(s)" | tee -a ${log_file}

  echo "Indexing - HNSW - ${data}"
  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  data_file="${pre_path}/${data}_base_pca.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_pca.index"
  log_file="./logger/${data}/PCA-HNSW-time.log"
  start_time=$(date +%s)
  ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW Index time: ${duration}(s)" | tee -a ${log_file}

  echo "Indexing -IVF- ${data}"
  if [ $data == "glove-200-angular" ]; then
     C=2176
  elif [ $data == "sift-128-euclidean" ]; then
     C=2000
  elif [ $data == "msong-420" ]; then
     C=1984
  elif [ $data == "contriever-768" ]; then
     C=1990
  elif [ $data == "gist-960-euclidean" ]; then
      C=2000
  elif [ $data == "deep-image-96-angular" ]; then
      C=12643
  elif [ $data == "instructorxl-arxiv-768" ]; then
      C=3002
  elif [ $data == "openai-1536-angular" ]; then
      C=1999
  elif [ $data == "glove-25-angular_100k" ]; then
      C=316
  elif [ $data == "glove-50-angular_100k" ]; then
      C=316
  elif [ $data == "glove-100-angular_100k" ]; then
      C=316
  elif [ $data == "glove-200-angular_100k" ]; then
      C=316
  elif [ $data == "glove-200-angular_1k" ]; then
      C=31
  elif [ $data == "glove-200-angular_10k" ]; then
      C=100
  elif [ $data == "instructorxl-arxiv-768_100k" ]; then
      C=316
  elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      C=31
  elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      C=100
  elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      C=1000
  fi
  log_file="./logger/${data}/PCA-IVF-time.log"
  start_time=$(date +%s)
  python3 ./DATA/ivfnew.py -d ${data} -m "pca" -k ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF Index time: ${duration}(s)" | tee -a ${log_file}

  data_file="${index_path}/${data}_base_pca.fvecs"
  centroid_file="${index_path}/${data}_centroid_pca.fvecs"
  index_file="${index_path}/${data}_ivf2_pca.index"
  adaptive=2
  log_file="./logger/${data}/PCA-HNSW-time.log"
  start_time=$(date +%s)
  ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF2 Index time: ${duration}(s)" | tee -a ${log_file}

  data_file="${index_path}/${data}_base_pca.fvecs"
  centroid_file="${index_path}/${data}_centroid_pca.fvecs"
  index_file="${index_path}/${data}_ivf1_pca.index"
  adaptive=1
  start_time=$(date +%s)
  ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF1 Index time: ${duration}(s)" | tee -a ${log_file}

done
