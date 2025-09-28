cd ..
source set.sh

pca_recall=0.995
# Learn Linear Model Without Negative Sample
data="instructorxl-arxiv-768_100k"

C_values=(158 316 474 632)
# C_values=(25 50 100 150 200)
M_values=(8 32)
efConstruction_values=(250 500 750)
# python3 ./DATA/pcanew.py -d ${data}
# python3 ./DATA/opq.py -d ${data}
index_path=./DATA/${data}
data_path=${store_path}/${data}
# for C in "${C_values[@]}"; do
#   # python3 ./DATA/ivf.py -d ${data} -m "naive" -k ${C}
#   # python3 ./DATA/ivf.py -d ${data} -m "O" -k ${C}
#   # for adaptive in {0..1}
#   #   do
#   #   index_path=./DATA/${data}

#   #   if [ $adaptive == "0" ] # raw vectors
#   #   then
#   #       data_file="${data_path}/${data}_base.fvecs"
#   #       centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
#   #       log_file="./logger/${data}/Naive-IVF-time.log"
#   #   else
#   #       data_file="${index_path}/O${data}_base.fvecs"
#   #       centroid_file="${index_path}/O${data}_centroid_${C}.fvecs"
#   #       log_file="./logger/${data}/ADS-IVF-time.log"
#   #   fi

#   #   # 0 - IVF, 1 - IVF++, 2 - IVF+
#   #   index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"
#   #   ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
#   # done

#   # echo "PCA - ${data}"
#   # python3 ./DATA/ivf.py -d ${data} -m "pca" -k ${C}

#   # data_file="${index_path}/${data}_base_pca.fvecs"
#   # centroid_file="${index_path}/${data}_centroid_pca_${C}.fvecs"
#   # index_file="${index_path}/${data}_ivf2_${C}_pca.index"
#   # adaptive=2
#   # ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

#   # data_file="${index_path}/${data}_base_pca.fvecs"
#   # centroid_file="${index_path}/${data}_centroid_pca_${C}.fvecs"
#   # index_file="${index_path}/${data}_ivf1_${C}_pca.index"
#   # adaptive=1
#   # ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

#   echo "OPQ - ${data}" 
#   python3 ./DATA/ivf.py -d ${data} -m "opq" -k ${C}

#   data_file="${index_path}/${data}_base_opq.fvecs"
#   centroid_file="${index_path}/${data}_centroid_opq_${C}.fvecs"
#   index_file="${index_path}/${data}_ivf_${C}_opq.index"
#   adaptive=0
#   ./build/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

# done
# # -----------------------------------------------------------------------------------------------------
for M in "${M_values[@]}"; do
  efConstruction=500
  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  # echo "Naive - HNSW - ${data}"
  # data_file="${data_path}/${data}_base.fvecs"
  # index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  # echo "ADS - HNSW - ${data}"
  # data_file="${pre_path}/O${data}_base.fvecs"
  # index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  # echo "PCA - HNSW - ${data}"
  # data_file="${pre_path}/${data}_base_pca.fvecs"
  # index_file="${index_path}/${data}_ef${efConstruction}_M${M}_pca.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  echo "OPQ - HNSW - ${data}"
  data_file="${pre_path}/${data}_base_opq.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
  ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done

for efConstruction in "${efConstruction_values[@]}"; do
  M=16

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  # echo "Naive - HNSW - ${data}"
  # data_file="${data_path}/${data}_base.fvecs"
  # index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  # echo "ADS - HNSW - ${data}"
  # data_file="${pre_path}/O${data}_base.fvecs"
  # index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  # echo "PCA - HNSW - ${data}"
  # data_file="${pre_path}/${data}_base_pca.fvecs"
  # index_file="${index_path}/${data}_ef${efConstruction}_M${M}_pca.index"
  # ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  echo "OPQ - HNSW - ${data}"
  data_file="${pre_path}/${data}_base_opq.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
  ./build/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done
# -----------------------------------------------------------------------------------------------------
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
# # Learn Linear Model With Negative Sample For HNSW
for K in 10; do
 for M in "${M_values[@]}"; do
   efConstruction=500
   echo "OPQ - HNSW - ${data}"
    if [ $data == "instructorxl-arxiv-768_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    fi
  
   data_path=${store_path}/${data}
   pre_data=./DATA/${data}

   index="${pre_data}/${data}_ef${efConstruction}_M${M}_opq.index"
   learn="${data_path}/${data}_learn.fvecs"
   ground="${data_path}/${data}_learn_groundtruth.ivecs"
   index_type="hnsw1"

   trans="${pre_data}/${data}_opq_matrix.fvecs"
   code_book="${pre_data}/${data}_codebook.centroid"
   linear="${pre_data}/linear/linear_${index_type}_ef${efConstruction}_M${M}_opq_${K}.log"

   logger="./logger/${data}_logger_opq_${index_type}_ef${efConstruction}_M${M}.fvecs"


   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
   python3 ./DATA/linearforindex.py -d ${data} -m "opq" -i ${index_type} -k ${K} -n ${M} -e ${efConstruction}
   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}



   trans="${pre_data}/${data}_pca_matrix.fvecs"
   index="${pre_data}/${data}_ef${efConstruction}_M${M}_pca.index"
   linear="${pre_data}/linear/linear_${index_type}_ef${efConstruction}_M${M}_pca_${K}.log"

   logger="./logger/${data}_logger_pca_${index_type}_ef${efConstruction}_M${M}.fvecs"

  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
  #  python3 ./DATA/linearforindex.py -d ${data} -m "pca" -i ${index_type} -k ${K} -n ${M} -e ${efConstruction}
  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

 done

done

for K in 10; do
 for efConstruction in "${efConstruction_values[@]}"; do
   M=16
   echo "efConstruction - HNSW - ${efConstruction}"
    if [ $data == "instructorxl-arxiv-768_100k" ]; then
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    fi
  
   data_path=${store_path}/${data}
   pre_data=./DATA/${data}

   index="${pre_data}/${data}_ef${efConstruction}_M${M}_opq.index"
   learn="${data_path}/${data}_learn.fvecs"
   ground="${data_path}/${data}_learn_groundtruth.ivecs"
   trans="${pre_data}/${data}_opq_matrix.fvecs"
   code_book="${pre_data}/${data}_codebook.centroid"

   index_type="hnsw1"
   linear="${pre_data}/linear/linear_${index_type}_ef${efConstruction}_M${M}_opq_${K}.log"

   logger="./logger/${data}_logger_opq_${index_type}_ef${efConstruction}_M${M}.fvecs"


   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
   python3 ./DATA/linearforindex.py -d ${data} -m "opq" -i ${index_type} -k ${K} -n ${M} -e ${efConstruction}
   ./build/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}



   trans="${pre_data}/${data}_pca_matrix.fvecs"
   index="${pre_data}/${data}_ef${efConstruction}_M${M}_pca.index"
   linear="${pre_data}/linear/linear_${index_type}_ef${efConstruction}_M${M}_pca_${K}.log"

   logger="./logger/${data}_logger_pca_${index_type}_ef${efConstruction}_M${M}.fvecs"

  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
  #  python3 ./DATA/linearforindex.py -d ${data} -m "pca" -i ${index_type} -k ${K} -n ${M} -e ${efConstruction}
  #  ./build/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

 done

done

# Learn Linear Model With Negative Sample For IVF

# for K in 10; do
#   for C in "${C_values[@]}"; do
#     echo "C - IVF - ${C}"
#     # efsearch <= nlist
#     if [ $data == "instructorxl-arxiv-768_100k" ]; then
#       efSearch=$C
#       opq_recall=0.995
#       pca_recall=0.995
#     fi

#     data_path=${store_path}/${data}
#     index_path=./DATA/${data}

#     index="${index_path}/${data}_ivf_${C}_opq.index"
#     linear="${index_path}/linear/linear_ivf_${C}_opq_${K}.log"
 
#     learn="${data_path}/${data}_learn.fvecs"
#     ground="${data_path}/${data}_learn_groundtruth.ivecs"
#     trans="${index_path}/${data}_opq_matrix.fvecs"
#     code_book="${index_path}/${data}_codebook.centroid"
#     logger="./logger/${data}_logger_opq_ivf_${C}.fvecs"


#     ./build/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
#     python3 ./DATA/linearforindex.py -d ${data} -m "opq" -i "ivf" -k ${K} -c ${C}
#     ./build/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}


#     index="${index_path}/${data}_ivf2_${C}_pca.index"
#     linear="${index_path}/linear/linear_ivf_${C}_pca_${K}.log"
 
#     trans="${index_path}/${data}_pca_matrix.fvecs"
#     logger="./logger/${data}_logger_pca_ivf_${C}.fvecs"

#     # ./build/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
#     # python3 ./DATA/linearforindex.py -d ${data} -m "pca" -i "ivf" -k ${K} -c ${C}
#     # ./build/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

#  done

# done
