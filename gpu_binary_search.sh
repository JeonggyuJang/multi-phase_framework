dev_num="3"
for prune_rate in 85
do
    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16_BN_IMAGE --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern micro --Method gpu --Weight_decay 0.0005 --gamma 0.2 --Dataset 100
done
#for prune_rate in 75
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16_BN_IMAGE --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern block --Method gpu --Weight_decay 0.0005 --gamma 0.2 --Dataset 100
#done
#for prune_rate in 85
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16_BN_IMAGE --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --method layer --Weight_decay 0.0005 --gamma 0.2 --Dataset 100
#done
# VGG16, Block_prune
#for prune_rate in 25
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model VGG16L_CIFAR_BIG --Train_Batch 64 --T_Epoch 220 --LR_step 80 160 180 200 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern micro --Method gpu --Weight_decay 0.0005 --gamma 0.2 --Dataset 100
#done
#for prune_rate in 50
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern micro --Method gpu
#done

# Block_top
#for prune_rate in 50 56.25 62.5 68.75
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern topbottom --Method gpu
#done

#for prune_rate in 75
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern pattern --Method gpu
#done
# BLock pruning
#for prune_rate in 43.75 
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern block --Method gpu
#done
#for prune_rate in 81.25 #87.5 93.75
#do
#    python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Dev $dev_num --momentum 0.9 --pattern block --Method gpu
#done
