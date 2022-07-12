dev_num="3"
#for prune_rate in 72.94 58.17
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16L_CIFAR_BIG --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --P_rate $prune_rate --lr 0.01 --Method kc --Dev $dev_num --with_fc t --Dataset 100 --Weight_decay 0.0005 --momentum 0.9 --gamma 0.2
#done

#for prune_rate in 35.32
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16L_CIFAR_BIG --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --P_rate $prune_rate --lr 0.01 --Method kc --Dev $dev_num --with_fc t --Dataset 100 --Weight_decay 0.0005 --momentum 0.9 --gamma 0.2
#done
#for prune_rate in 75
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16L_CIFAR_BIG --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.001 --Method kc --Dev $dev_num --with_fc t --Dataset 100 --Weight_decay 0.0005 --momentum 0.9 --gamma 0.2
#done
##### IMAGENET!!!!!
for prune_rate in 75
do
    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16_BN_IMAGE --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.001 --Method kc --Dev $dev_num --with_fc t --Dataset 100 --Weight_decay 0.0005 --momentum 0.9 --gamma 0.2
done
