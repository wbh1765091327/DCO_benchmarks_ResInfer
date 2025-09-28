cd ..
source set.sh

pca_recall=0.995
# Learn Linear Model Without Negative Sample
# datasets=(
#   # "glove-25-angular_100k"
#   # "glove-50-angular_100k"
#   # "glove-100-angular_100k"
#   # "glove-200-angular_100k"
# #   "glove-200-angular_1k"
#   "glove-200-angular_10k"
# )
# datasets=(
#     "instructorxl-arxiv-768_1000k"
# )
for data in "${datasets[@]}"; do
  echo "precompute - ${data}"
  for K in 10; do
    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    base="${index_path}/${data}_base_pca.fvecs"
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"
    trans="${index_path}/${data}_pca_matrix.fvecs"
    linear="${index_path}/linear/linear_${K}_l2.log"
    ./build/src/binery_search_parameter -d 0 -n $base -q $learn -g $ground -t $trans -l $linear -k $K -e $pca_recall

    linear="${index_path}/linear/linear_${K}_ip.log"
    ./build/src/binery_search_parameter -d 1 -n $base -q $learn -g $ground -t $trans -l $linear -k $K -e $pca_recall
  done
done

# Learn Linear Model With Negative Sample For HNSW
for K in 10; do
 for data in "${datasets[@]}"; do
   echo "Indexing - ${data}"
   if [ $data == "nytimes-16-angular" ]; then
      efSearch=500
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-50-angular" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "sift-128-euclidean" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "msong-420" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "contriever-768" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "gist-960-euclidean" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "deep-image-96-angular" ]; then
      efSearch=2000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768" ]; then
      efSearch=2000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "openai-1536-angular" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-25-angular_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-50-angular_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-100-angular_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-100-angular_10k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular_1k" ]; then
      efSearch=100
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      efSearch=100
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    fi
  

   data_path=${store_path}/${data}
   pre_data=./DATA/${data}

   index="${pre_data}/${data}_ef500_M16_opq.index"
   learn="${data_path}/${data}_learn.fvecs"
   ground="${data_path}/${data}_learn_groundtruth.ivecs"
   trans="${pre_data}/${data}_opq_matrix.fvecs"
   code_book="${pre_data}/${data}_codebook.centroid"

   index_type="hnsw1"
   linear="${pre_data}/linear/linear_${index_type}_opq_${K}.log"
   logger="./logger/${data}_logger_opq_${index_type}.fvecs"

    log_file="./logger/${data}/OPQ-HNSW-Linear-time.log"
    start_time=$(date +%s)
   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}

   python3 ./DATA/linear.py -d ${data} -m "opq" -i ${index_type} -k ${K}

   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "OPQ HNSW Linear with ${K} time: ${duration}(s)" | tee -a ${log_file}


   trans="${pre_data}/${data}_pca_matrix.fvecs"
   index="${pre_data}/${data}_ef500_M16_pca.index"
   linear="${pre_data}/linear/linear_${index_type}_pca_${K}.log"
   logger="./logger/${data}_logger_pca_${index_type}.fvecs"

  #   log_file="./logger/${data}/PCA-HNSW-Linear-time.log"
  #   start_time=$(date +%s)
  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

  #  python3 ./DATA/linear.py -d ${data} -m "pca" -i ${index_type} -k ${K}

  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
  #   end_time=$(date +%s)
  #   duration=$((end_time - start_time))
  #   echo "PCA HNSW Linear with ${K} time: ${duration}(s)" | tee -a ${log_file}
 done

done

# Learn Linear Model With Negative Sample For IVF

for K in 10; do
  for data in "${datasets[@]}"; do
    echo "Indexing - ${data}"
    # efsearch <= nlist
    if [ $data == "glove-200-angular" ]; then
      efSearch=2176
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "sift-128-euclidean" ]; then
      efSearch=2000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "msong-420" ]; then
      efSearch=1984
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "contriever-768" ]; then
      efSearch=1990
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "gist-960-euclidean" ]; then
      efSearch=2000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "deep-image-96-angular" ]; then
      efSearch=12643
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768" ]; then
      efSearch=3002
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "openai-1536-angular" ]; then
      efSearch=1999
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-25-angular_100k" ]; then
      efSearch=316
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-50-angular_100k" ]; then
      efSearch=316
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-100-angular_100k" ]; then
      efSearch=316
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular_100k" ]; then
      efSearch=316
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular_1k" ]; then
      efSearch=31
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "glove-200-angular_10k" ]; then
      efSearch=100
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_100k" ]; then
      efSearch=316
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      efSearch=31
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      efSearch=100
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}

    index="${index_path}/${data}_ivf_opq.index"
    linear="${index_path}/linear/linear_ivf_opq_${K}.log"
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"
    trans="${index_path}/${data}_opq_matrix.fvecs"
    code_book="${index_path}/${data}_codebook.centroid"
    logger="./logger/${data}_logger_opq_ivf.fvecs"

    log_file="./logger/${data}/OPQ-IVF-Linear-time.log"
    start_time=$(date +%s)
    ./build/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}

    python3 ./DATA/linear.py -d ${data} -m "opq" -i "ivf" -k ${K}

    ./build/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "OPQ IVF Linear with ${K} time: ${duration}(s)" | tee -a ${log_file}


    index="${index_path}/${data}_ivf2_pca.index"
    linear="${index_path}/linear/linear_ivf_pca_${K}.log"
    trans="${index_path}/${data}_pca_matrix.fvecs"
    logger="./logger/${data}_logger_pca_ivf.fvecs"

    # log_file="./logger/${data}/PCA-IVF-Linear-time.log"
    # start_time=$(date +%s)
    # ./build/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

    # python3 ./DATA/linear.py -d ${data} -m "pca" -i "ivf" -k ${K}

    # ./build/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
    # end_time=$(date +%s)
    # duration=$((end_time - start_time))
    # echo "PCA IVF Linear with ${K} time: ${duration}(s)" | tee -a ${log_file}
 done

done
