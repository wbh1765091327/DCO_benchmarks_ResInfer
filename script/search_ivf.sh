# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0
sigma=8
efSearch=50
C=4096
cd ..

source set.sh
# datasets=(
#     # "glove-200-angular"
#     # "msong-420"
#     # "gist-960-euclidean"
#     # "deep-image-96-angular"
#     # "contriever-768"
#     # "instructorxl-arxiv-768"
#     # "openai-1536-angular"
#     # "sift-128-euclidean"
# )
# datasets=(
#   "glove-25-angular_100k"
#   "glove-50-angular_100k"
#   "glove-100-angular_100k"
#   "glove-200-angular_100k"
# )
for K in 10; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "glove-200-angular" ]; then
      C=2176
      sigma=16
    elif [ $data == "sift-128-euclidean" ]; then
      C=2000
      sigma=8
    elif [ $data == "msong-420" ]; then
      C=1984
      sigma=12
    elif [ $data == "contriever-768" ]; then
      C=1990
      sigma=12
    elif [ $data == "gist-960-euclidean" ]; then
      C=2000
      sigma=8
    elif [ $data == "deep-image-96-angular" ]; then
      C=12643
      sigma=8
    elif [ $data == "instructorxl-arxiv-768" ]; then
      C=3002
      sigma=12
    elif [ $data == "openai-1536-angular" ]; then
      C=1999
      sigma=16
    elif [ $data == "glove-25-angular_100k" ]; then
      C=316
      sigma=10
      delta_d=16
    elif [ $data == "glove-50-angular_100k" ]; then
      C=316
      sigma=12
      delta_d=32
    elif [ $data == "glove-100-angular_100k" ]; then
      C=316
      sigma=14
      delta_d=32
    elif [ $data == "glove-200-angular_100k" ]; then
      C=316
      sigma=16
      delta_d=32
    elif [ $data == "glove-200-angular_10k" ]; then
      C=100
      sigma=16
      delta_d=32
    elif [ $data == "glove-200-angular_1k" ]; then
      C=31
      sigma=16
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      C=31
      sigma=12
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      C=100
      sigma=12
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_100k" ]; then
      C=316
      sigma=12
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      C=1000
      sigma=12
      delta_d=32
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/ivf/nosimd/${data}"
    
    # 自动创建结果目录
    mkdir -p ${result_path}

    # for randomize in {0..1}; do
    #   if [ $randomize == "1" ]; then
    #     echo "IVF++"
    #   elif [ $randomize == "2" ]; then
    #     echo "IVF+"

    #   else
    #     echo "IVF"
    #   fi

    #   # res="${result_path}/${data}_ad_ivf_${randomize}.log"
    #   res1="${result_path}/${data}_ad_ivf_${randomize}_dist_time.log"
    #   res2="${result_path}/${data}_ad_ivf_${randomize}_pruning_time.log"
    #   index="${index_path}/${data}_ivf_${C}_${randomize}.index"

    #   query="${data_path}/${data}_query.fvecs"
    #   gnd="${data_path}/${data}_groundtruth.ivecs"
    #   trans="${index_path}/O.fvecs"
    #   ./build/src/search_ivf_dist -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -k ${K} -s ${efSearch}
    #   # ./build/src/search_ivf_pruning -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -k ${K} -s ${efSearch}

    # done
    
    # # DDC-pca
    # randomize=3
    # res1="${result_path}/${data}_ad_ivf_${randomize}_dist_time.log"
    # res2="${result_path}/${data}_ad_ivf_${randomize}_pruning_time.log"
    # index="${index_path}/${data}_ivf1_pca.index"
    # query="${data_path}/${data}_query.fvecs"
    # gnd="${data_path}/${data}_groundtruth.ivecs"
    # trans="${index_path}/${data}_pca_matrix.fvecs"
    # linear="${index_path}/linear/linear_ivf_pca_${K}.log"
    # ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
    # # ./build/src/search_ivf_pruning -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

    # # DDC-res
    # randomize=5
    # res1="${result_path}/${data}_ad_ivf_${randomize}_dist_time.log"
    # res2="${result_path}/${data}_ad_ivf_${randomize}_pruning_time.log"
    # index="${index_path}/${data}_ivf1_pca.index"
    # query="${data_path}/${data}_query.fvecs"
    # gnd="${data_path}/${data}_groundtruth.ivecs"
    # trans="${index_path}/${data}_pca_matrix_${K}.fvecs"
    # ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch}
    # # ./build/src/search_ivf_pruning -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch}

    # DDC-opq
    randomize=6
    res="${result_path}/${data}_ad_ivf_${randomize}_dist_time.log"
    index="${index_path}/${data}_ivf_opq.index"
    code="${index_path}/${data}_codebook.centroid"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${index_path}/${data}_opq_matrix.fvecs"
    linear="${index_path}/linear/linear_ivf_opq_${K}.log"
    ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
    #  ./build/src/search_ivf_pruning -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
  done
done
