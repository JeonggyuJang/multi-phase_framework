dev_num="0"

#CIFAR100
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 2048 --T_Epoch 1000 --LR_step 200 --lr 0.02 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 50  --Model VGG16_CIFAR_BIG_drop_out
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 400 --LR_step 100 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-50 --dP_rate 0  --Model VGG16_CIFAR_BIG 
