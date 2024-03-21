docker rm -f dit_train 2>/dev/null

# 做一些检测
#docker run -ti --gpus=all --rm dit_runtime:v1 nvidia-smi
#docker run -ti --gpus=all --rm dit_runtime:v1 python -c "import torch; print(torch.__file__)"


# 1. docker train 必须使用前台
# 2. 需要宿主机的 libcudart.so.{宿主机_cuda_version}, 一般在 /usr/local/cuda/lib64/libcudart.so.xxx
docker run -ti --security-opt seccomp=unconfined \
--net=host \
--gpus=all \
--ipc=host \
-v $PWD/object_detection:/ws/object_detection \
-v /usr/local/cuda/lib64/libcudart.so.12:/usr/local/cuda/lib64/libcudart.so.12 \
--name dit_train \
dit_runtime:v1 \
	bash object_detection/00_train.sh
