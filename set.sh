# export datasets=("gist" "deep1M" "_glove2.2m" "_tiny5m" "_sift10m" "_word2vec" "_msong")
#export datasets=("deep1M" "_msong" "_glove2.2m" "_tiny5m" "_word2vec" "gist")
#export datasets=("sift20m" "sift40m" "sift60m" "sift80m" "sift100m")
# export datasets=("glove-200-angular" "sift-128-euclidean" "msong-420" "contriever-768" "gist-960-euclidean" "deep-image-96-angular" "instructorxl-arxiv-768" "openai-1536-angular")
# export datasets=("glove-200-angular" "sift-128-euclidean" "contriever-768" "gist-960-euclidean")
# export datasets=("sift-128-euclidean" "msong-420" "contriever-768" "deep-image-96-angular" "instructorxl-arxiv-768" "openai-1536-angular")
# export datasets=("glove-200-angular_100k")
export datasets=("instructorxl-arxiv-768_1k" "instructorxl-arxiv-768_10k" "instructorxl-arxiv-768_1000k")
# export datasets=("glove-25-angular_100k" "glove-50-angular_100k" "glove-100-angular_100k" "glove-200-angular_100k" "glove-200-angular_1k" "glove-200-angular_10k")
# export datasets=("glove-200-angular_1k" "glove-200-angular_10k" "glove-200-angular_100k")
# export datasets=("glove-50-angular_100k" "glove-100-angular_100k" "glove-200-angular_100k" "glove-200-angular_1k" "glove-200-angular_10k")
export store_path=$HOME/cppwork/Res-Infer/DATA
# the operation to determine use SSE define in ./src/search_hnsw.cpp ./src/search_ivf.cpp