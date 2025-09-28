
efSearch=50
sigma=8
delta_d=32
finger_lsh_dim=64
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
# )
datasets=(
    "glove-25-angular_100k"
)
for K in 10; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "nytimes-16-angular" ]; then
      sigma=14
      delta_d=4
      finger_lsh_dim=8
    elif [ $data == "glove-50-angular" ]; then
      sigma=12
      delta_d=12
      finger_lsh_dim=32
    elif [ $data == "glove-200-angular" ]; then
      sigma=16
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "sift-128-euclidean" ]; then
      sigma=8
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "msong-420" ]; then
      sigma=12
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "contriever-768" ]; then
      sigma=12
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "gist-960-euclidean" ]; then
      sigma=8
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "deep-image-96-angular" ]; then
      sigma=8
      delta_d=24
      finger_lsh_dim=64
    elif [ $data == "instructorxl-arxiv-768" ]; then
      sigma=12
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "openai-1536-angular" ]; then
      sigma=16
      delta_d=32
      finger_lsh_dim=64
    elif [ $data == "glove-25-angular_100k" ]; then
      sigma=10
      delta_d=16
      finger_lsh_dim=16
    elif [ $data == "glove-50-angular_100k" ]; then
      sigma=12
      delta_d=16
      finger_lsh_dim=16
    elif [ $data == "glove-100-angular_100k" ]; then
      sigma=14
      delta_d=16
      finger_lsh_dim=16
    elif [ $data == "glove-200-angular_100k" ]; then
      sigma=16
      delta_d=16
      finger_lsh_dim=16
    fi
    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    ef=500
    M=16
    randomize=9
    echo "HNSW"
    index="${index_path}/${data}_ef${ef}_M${M}.index"
    # res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    result_path1="./results/recall@${K}/hnsw/finger-avx512/${data}"
    result_path2="./results/recall@${K}/hnsw/finger-nosimd/${data}"
    mkdir -p ${result_path1} ${result_path2}
    res1="${result_path1}/${data}_ad_hnsw_${randomize}_dist_time.log"
    res2="${result_path1}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    res3="${result_path2}/${data}_ad_hnsw_${randomize}_dist_time.log"
    res4="${result_path2}/${data}_ad_hnsw_${randomize}_pruning_stats.log"
    
    # ./build/src/search_hnsw_avx512_dist -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res1} -k ${K}  -n ${data}
    # ./build/src/search_hnsw_avx512_pruning -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res2} -k ${K}  -n ${data}
    ./build/src/search_hnsw_dist -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res3} -k ${K}  -n ${data}
    # ./build/src/search_hnsw_pruning -d ${randomize} -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res4} -k ${K}  -n ${data}
 
  done
done
