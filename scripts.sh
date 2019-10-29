# cifar10_32x32_res50

nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 50 --model-prefix models/cifar-res50 \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,32,32 --data-nthreads 8 --save-period 5 &
tail -f nohup.out

nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 50 --model-prefix models/cifar-res50-mish \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,32,32 --data-nthreads 8 --save-period 5 --act-type mish &
tail -f nohup.out

nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 50 --model-prefix models/cifar-res50-mish \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,32,32 --data-nthreads 8 --save-period 5 --act-type nish &
tail -f nohup.out

## cifar10_28x28_res32

nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 32 --model-prefix models/cifar-res32-gish \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,28,28 --data-nthreads 8 --save-period 5  &
tail -f nohup.out

nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 32 --model-prefix models/cifar-res32-gish \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,28,28 --data-nthreads 8 --save-period 5 --act-type mish &
tail -f nohup.out


nohup python train_cifar10.py --gpus 0,1 --network resnet --num-layers 32 --model-prefix models/cifar-res32-gish \
    --max-random-aspect-ratio 0.1 --max-random-rotate-angle 15 --max-random-shear-ratio 0.1 \
    --resize 32 --image-shape 3,28,28 --data-nthreads 8 --save-period 5 --act-type nish &
tail -f nohup.out

