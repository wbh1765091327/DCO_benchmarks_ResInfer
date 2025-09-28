
efSearch=50
sigma=8
delta_d=32
cd ..

source set.sh
# datasets=(
#     "glove-200-angular"
#     "msong-420"
#     "gist-960-euclidean"
#     "deep-image-96-angular"
#     "contriever-768"
#     "instructorxl-arxiv-768"
#     "openai-1536-angular"
#     "sift-128-euclidean"
# )

datasets=(
    "deep-image-96-angular"
)
for K in 10; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "nytimes-16-angular" ]; then
      efSearch=50
      sigma=14
      delta_d=4
    elif [ $data == "glove-50-angular" ]; then
      efSearch=100
      sigma=12
      delta_d=12
    elif [ $data == "glove-200-angular" ]; then
      efSearch=500
      sigma=16
      delta_d=32
    elif [ $data == "sift-128-euclidean" ]; then
      efSearch=100
      sigma=8
      delta_d=32
    elif [ $data == "msong-420" ]; then
      efSearch=100
      sigma=12
      delta_d=32
    elif [ $data == "contriever-768" ]; then
      efSearch=10
      sigma=12
      delta_d=32
    elif [ $data == "gist-960-euclidean" ]; then
      efSearch=100
      sigma=8
      delta_d=32
    elif [ $data == "deep-image-96-angular" ]; then
      efSearch=100
      sigma=8
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768" ]; then
      efSearch=100
      sigma=12
      delta_d=32
    elif [ $data == "openai-1536-angular" ]; then
      efSearch=100
      sigma=16
      delta_d=32
    elif [ $data == "glove-25-angular_100k" ]; then
      efSearch=1000
      sigma=10
      delta_d=16
    elif [ $data == "glove-50-angular_100k" ]; then
      efSearch=1000
      sigma=12
      delta_d=32
    elif [ $data == "glove-100-angular_100k" ]; then
      efSearch=1000
      sigma=14
      delta_d=32
    elif [ $data == "glove-200-angular_100k" ]; then
      efSearch=1000
      sigma=16
      delta_d=32
    elif [ $data == "glove-200-angular_1k" ]; then
      efSearch=100
      sigma=16
      delta_d=32
    elif [ $data == "glove-200-angular_10k" ]; then
      efSearch=100
      sigma=16
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_1k" ]; then
      efSearch=100
      sigma=12
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_10k" ]; then
      efSearch=100
      sigma=12
      delta_d=32
    elif [ $data == "instructorxl-arxiv-768_1000k" ]; then
      efSearch=100
      sigma=12
      delta_d=32
    fi
    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/hnsw/nosimd/${data}"
    temp_data=./DATA/${data}
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    ef=500
    M=16
    # 自动创建结果目录
    mkdir -p ${result_path}

    # for randomize in {0..1}; do
    #   if [ $randomize == "1" ]; then
    #     echo "HNSW++"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"
    #   elif [ $randomize == "2" ]; then
    #     echo "HNSW+"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"dwdw
    #   else
    #     echo "HNSW"
    #     index="${index_path}/${data}_ef${ef}_M${M}.index"
    #   fi

    #   res1="${result_path}/${data}_ad_hnsw_${randomize}_dist_time.log"
    #   res2="${result_path}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    #   trans="${temp_data}/O.fvecs"
    #   ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -k ${K} -s ${efSearch}
     
    # done

    # index="${index_path}/${data}_ef500_M16_pca.index"
    # trans="${temp_data}/${data}_pca_matrix.fvecs"
    # randomize=6
    # res1="${result_path}/${data}_ad_hnsw_${randomize}_dist_time.log"
    # res2="${result_path}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    # linear="${index_path}/linear/linear_hnsw1_pca_${K}.log"
    # ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
    

    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix_${K}.fvecs"
    randomize=7
    res1="${result_path}/${data}_ad_hnsw_${randomize}_dist_time.log"
    res2="${result_path}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch}
    
    # index="${index_path}/${data}_ef500_M16_opq.index"
    # trans="${temp_data}/${data}_opq_matrix.fvecs"
    # randomize=3
    # code="${temp_data}/${data}_codebook.centroid"
    # res1="${result_path}/${data}_ad_hnsw_${randomize}_dist_time.log"
    # res2="${result_path}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    # linear="${index_path}/linear/linear_hnsw1_opq_${K}.log"
    # ./build/src/search_hnsw_dist -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
    # # ./cmake-build-debug/src/search_hnsw_pruning -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
  done
done
