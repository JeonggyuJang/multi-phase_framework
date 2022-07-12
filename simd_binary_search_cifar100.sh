dev_num="1"
for prune_rate in 82.5
do
    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16L_CIFAR_BIG --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --P_rate $prune_rate --lr 0.01 --Method simd --Dev $dev_num --with_fc t --Dataset 100 --Weight_decay 0.0005 --momentum 0.9 --gamma 0.2
done
