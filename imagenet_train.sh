#KC pruning for imagenet with single-gpu
#python ./imagenet_training_framework.py --arch vgg16_bn --epochs 30 --batch-size 256 --lr 0.01 --momentum 0.9 /data/ILSVRC/ILSVRC/Data/CLS-LOC
#python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 60 --batch-size 256 --lr 0.01 --momentum 0.9 /data/ILSVRC/ILSVRC/Data/

#KC pruning for imagenet with multi-gpu
#python ./imagenet_training_framework.py --arch vgg16_bn --epochs 40 --resume ./checkpoint/checkpoint.pth.tar --batch-size 256 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data
#python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 40 --resume ./checkpoint/checkpoint_gpu.pth.tar --batch-size 220 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data

# BLOCK
#python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 35 --batch-size 220 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data
python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 35 --resume ./checkpoint/checkpoint_gpu.pth.tar --batch-size 200 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data

# LAYER
#python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 25 --batch-size 200 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data
#python ./imagenet_training_framework_gpu.py --arch vgg16_bn --epochs 35 --resume ./checkpoint/checkpoint_gpu.pth.tar --batch-size 200 --lr 0.01 --momentum 0.9 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/ILSVRC/ILSVRC/Data
