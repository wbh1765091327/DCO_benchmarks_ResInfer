
efSearch=50
sigma=8
delta_d=32
cd ..

source set.sh
data="instructorxl-arxiv-768_100k"

# C_values=(25 50 100 150 200)
C_values=(158 316 632)
M_values=(8 32)
efConstruction_values=(250 500 750)

for K in 10; do
  for M in "${M_values[@]}"; do
    ef=500
    echo "Searching - ${data}"
    if [ $data == "instructorxl-arxiv-768_100k" ]; then
      sigma=12
      delta_d=32
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path_simd="./results/recall@${K}/hnsw/simd/${data}"
    result_path_nosimd="./results/recall@${K}/hnsw/nosimd/${data}"
    temp_data=./DATA/${data}
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"


    # for randomize in {0..1}; do
    #   if [ $randomize == "1" ]; then
    #     echo "HNSW++"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"
    #   elif [ $randomize == "2" ]; then
    #     echo "HNSW+"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"
    #   else
    #     echo "HNSW"
    #     index="${index_path}/${data}_ef${ef}_M${M}.index"
    #   fi

    #   res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    #   res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    #   trans="${temp_data}/O.fvecs"
    #   # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -k ${K} -s ${efSearch} 
    #   ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -k ${K} -s ${efSearch} 
      
    # done

    # index="${index_path}/${data}_ef${ef}_M${M}_pca.index"
    # trans="${temp_data}/${data}_pca_matrix.fvecs"
    # randomize=6
    # res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # linear="${index_path}/linear/linear_hnsw1_ef${ef}_M${M}_pca_${K}.log"
    # echo "HNSW" ${linear}
    # # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    # ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    

    # index="${index_path}/${data}_ef${ef}_M${M}_pca.index"
    # trans="${temp_data}/${data}_pca_matrix_${K}.fvecs"
    # randomize=7
    # res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} 
    # ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} 

    index="${index_path}/${data}_ef${ef}_M${M}_opq.index"
    trans="${temp_data}/${data}_opq_matrix.fvecs"
    randomize=4
    code="${temp_data}/${data}_codebook.centroid"
    res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_simd_dist_time.log"
    res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    linear="${index_path}/linear/linear_hnsw1_ef${ef}_M${M}_opq_${K}.log"
    # ./build/src/search_hnsw_avx512_dist -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    ./build/src/search_hnsw_dist -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
  done
done

for K in 10; do
  for ef in "${efConstruction_values[@]}"; do
    M=16
    echo "Searching - ${data}"
    if [ $data == "instructorxl-arxiv-768_100k" ]; then
      sigma=12
      delta_d=32
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path_simd="./results/recall@${K}/hnsw/simd/${data}"
    result_path_nosimd="./results/recall@${K}/hnsw/nosimd/${data}"
    temp_data=./DATA/${data}
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"


    # for randomize in {0..1}; do
    #   if [ $randomize == "1" ]; then
    #     echo "HNSW++"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"
    #   elif [ $randomize == "2" ]; then
    #     echo "HNSW+"
    #     index="${index_path}/O${data}_ef${ef}_M${M}.index"
    #   else
    #     echo "HNSW"
    #     index="${index_path}/${data}_ef${ef}_M${M}.index"
    #   fi

    #   res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    #   res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    #   trans="${temp_data}/O.fvecs"
    #   # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -k ${K} -s ${efSearch} 
    #   ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -k ${K} -s ${efSearch} 
      
    # done

    # index="${index_path}/${data}_ef${ef}_M${M}_pca.index"
    # trans="${temp_data}/${data}_pca_matrix.fvecs"
    # randomize=6
    # res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # linear="${index_path}/linear/linear_hnsw1_ef${ef}_M${M}_pca_${K}.log"
    # # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    # ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    

    # index="${index_path}/${data}_ef${ef}_M${M}_pca.index"
    # trans="${temp_data}/${data}_pca_matrix_${K}.fvecs"
    # randomize=7
    # res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    # # ./build/src/search_hnsw_avx512_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} 
    # ./build/src/search_hnsw_dist -d ${randomize} -p ${delta_d} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} 

    index="${index_path}/${data}_ef${ef}_M${M}_opq.index"
    trans="${temp_data}/${data}_opq_matrix.fvecs"
    randomize=4
    code="${temp_data}/${data}_codebook.centroid"
    res1="${result_path_nosimd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_simd_dist_time.log"
    res2="${result_path_simd}/${data}_ad_hnsw_ef${ef}_M${M}_${randomize}_dist_time.log"
    linear="${index_path}/linear/linear_hnsw1_ef${ef}_M${M}_opq_${K}.log"
    # ./build/src/search_hnsw_avx512_dist -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
    ./build/src/search_hnsw_dist -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} 
  done
done

# for K in 10; do
#   for C in "${C_values[@]}"; do
#     echo "Searching - ${data}"

#     if [ $data == "instructorxl-arxiv-768_100k" ]; then
#       sigma=12
#       delta_d=32
#     fi

#     data_path=${store_path}/${data}
#     index_path=./DATA/${data}
#     result_path_nosimd="./results/recall@${K}/ivf/nosimd/${data}"
#     result_path_simd="./results/recall@${K}/ivf/simd/${data}"

#     # for randomize in {0..1}; do
#     #   if [ $randomize == "1" ]; then
#     #     echo "IVF++"
#     #   elif [ $randomize == "2" ]; then
#     #     echo "IVF+"

#     #   else
#     #     echo "IVF"
#     #   fi

#     #   # res="${result_path}/${data}_ad_ivf_${randomize}.log"
#     #   res1="${result_path_nosimd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     #   res2="${result_path_simd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     #   index="${index_path}/${data}_ivf_${C}_${randomize}.index"

#     #   query="${data_path}/${data}_query.fvecs"
#     #   gnd="${data_path}/${data}_groundtruth.ivecs"
#     #   trans="${index_path}/O.fvecs"
#     #   # ./build/src/search_ivf_dist -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -k ${K} -s ${efSearch}
#     #   ./build/src/search_ivf_avx512_dist -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -k ${K} -s ${efSearch}

#     # done
    
#     # # DDC-pca
#     # randomize=3
#     # res1="${result_path_nosimd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     # res2="${result_path_simd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     # index="${index_path}/${data}_ivf1_${C}_pca.index"
#     # query="${data_path}/${data}_query.fvecs"
#     # gnd="${data_path}/${data}_groundtruth.ivecs"
#     # trans="${index_path}/${data}_pca_matrix.fvecs"
#     # linear="${index_path}/linear/linear_ivf_${C}_pca_${K}.log"
#     # ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
#     # # ./build/src/search_ivf_avx512_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

#     # # DDC-res
#     # randomize=5
#     # res1="${result_path_nosimd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     # res2="${result_path_simd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     # index="${index_path}/${data}_ivf1_${C}_pca.index"
#     # query="${data_path}/${data}_query.fvecs"
#     # gnd="${data_path}/${data}_groundtruth.ivecs"
#     # trans="${index_path}/${data}_pca_matrix_${K}.fvecs"
#     # # ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch}
#     # ./build/src/search_ivf_avx512_dist -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch}

#     # DDC-opq
#     randomize=6
#     res1="${result_path_nosimd}/${data}_ad_ivf_${C}_${randomize}_simd_dist_time.log"
#     res2="${result_path_simd}/${data}_ad_ivf_${C}_${randomize}_dist_time.log"
#     index="${index_path}/${data}_ivf_${C}_opq.index"
#     code="${index_path}/${data}_codebook.centroid"
#     query="${data_path}/${data}_query.fvecs"
#     gnd="${data_path}/${data}_groundtruth.ivecs"
#     trans="${index_path}/${data}_opq_matrix.fvecs"
#     linear="${index_path}/linear/linear_ivf_${C}_opq_${K}.log"
#     # ./build/src/search_ivf_dist -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res1} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
#     ./build/src/search_ivf_avx512_dist -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res2} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}
#   done
# done