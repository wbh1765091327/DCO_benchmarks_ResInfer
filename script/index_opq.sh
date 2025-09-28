cd ..
source set.sh

efConstruction=500
M=16
# datasets=(
#   # "glove-25-angular_100k"
#   # "glove-50-angular_100k"
#   # "glove-100-angular_100k"
#   # "glove-200-angular_100k"
# #   "glove-200-angular_1k"
#   "glove-200-angular_10k"
# )
for data in "${datasets[@]}"; do

  echo "precompute - ${data}"
  log_file="./logger/${data}/OPQ-Train-time.log"
  start_time=$(date +%s)
  python3 ./DATA/opq.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "OPQ Train time: ${duration}(s)" | tee -a ${log_file}

  echo "Indexing - HNSW - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  data_file="${pre_path}/${data}_base_opq.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
  log_file="./logger/${data}/OPQ-HNSW-time.log"
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
  elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      C=31
  elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      C=100
  elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      C=1000
  fi
  python3 ./DATA/ivfnew.py -d ${data} -m "opq" -k ${C}

  data_file="${index_path}/${data}_base_opq.fvecs"
  centroid_file="${index_path}/${data}_centroid_opq.fvecs"
  index_file="${index_path}/${data}_ivf_opq.index"
  adaptive=0

  log_file="./logger/${data}/OPQ-IVF-time.log"
  start_time=$(date +%s)
  ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF Index time: ${duration}(s)" | tee -a ${log_file}

done
