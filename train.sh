dev_num="1"

#python multi_phase_network_framework.py --T 1 --P 0 --R 0 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 80 140 170 190  --lr 0.02 --with_fc t --Dev $dev_num --Vis f --momentum 0.9
python multi_phase_network_framework.py --T 1 --P 0 --R 0 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 300 --LR_step 80 140 180 220 260  --lr 0.1 --with_fc t --Dev $dev_num --Vis f --momentum 0.9

