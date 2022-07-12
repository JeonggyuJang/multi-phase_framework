dev_num="0"

#for prune_rate in 10 20 30 40 50 60
#for prune_rate in 84.375
for prune_rate in 99
do
    #python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --T_Epoch 250 --LR_step 50 100 150 200 --P_rate $prune_rate --lr 0.01 --Method simd --Dev $dev_num --with_fc t --momentum 0.9 --gamma 0.1 --Dataset 100
    #python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --T_Epoch 250 --LR_step 50 100 150 200 --P_rate $prune_rate --lr 0.01 --Method simd --Dev $dev_num --with_fc t --momentum 0.9 --gamma 0.1 --Dataset 10
    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --T_Epoch 250 --LR_step 50 100 150 200 --P_rate $prune_rate --lr 0.01 --Method simd --Dev $dev_num --with_fc t --momentum 0.5 --gamma 0.1 --Dataset 10
done

#for prune_rate in 0
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --P_rate $prune_rate --lr 0.01 --Method simd --Dev $dev_num --with_fc t --momentum 0.9 --Meta 1
#done

