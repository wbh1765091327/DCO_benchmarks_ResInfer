cd ..
source set.sh
C=4096
pca_dim=32
opq_dim=96
# datasets=(
#   # "glove-25-angular_100k"
#   # "glove-50-angular_100k"
#   # "glove-100-angular_100k"
#   # "glove-200-angular_100k"
# #   "glove-200-angular_1k"
#   "glove-200-angular_10k"
# )
for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"
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
  log_file="./logger/${data}/Naive-IVF-time.log"
  start_time=$(date +%s)
  python3 ./DATA/ivfnew.py -d ${data} -m "naive" -k ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "cluster time: ${duration}(s)" | tee -a ${log_file}

  log_file="./logger/${data}/ADS-IVF-time.log"
  start_time=$(date +%s)
  python3 ./DATA/ivfnew.py -d ${data} -m "O" -k ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "cluster time: ${duration}(s)" | tee -a ${log_file}


  for adaptive in {0..1}
  do

  echo "Indexing - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}

  if [ $adaptive == "0" ] # raw vectors
  then
      data_file="${data_path}/${data}_base.fvecs"
      centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
      log_file="./logger/${data}/Naive-IVF-time.log"
  else
      data_file="${index_path}/O${data}_base.fvecs"
      centroid_file="${index_path}/O${data}_centroid_${C}.fvecs"
      log_file="./logger/${data}/ADS-IVF-time.log"
  fi

  # 0 - IVF, 1 - IVF++, 2 - IVF+
  index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"

  start_time=$(date +%s)
  ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF Index time: ${duration}(s)" | tee -a ${log_file}
    done
  done
