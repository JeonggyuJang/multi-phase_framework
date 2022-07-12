dev_num="1"

#for prune_rate in 10 20 30 40
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model ResNet18 --Train_Batch 64 --Dataset 10 --T_Epoch 10 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9
#done

#for prune_rate in 93.71 95.38 
#for prune_rate in 10 20 30 40
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --Dataset 10 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9
#done
# For VGG7 Network : High LR - Low LR
#for prune_rate in 91.6765
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --Dataset 10 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9 --Meta 1 
#done
#for prune_rate in 86.428
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --Dataset 10 --T_Epoch 180 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9
#done
#-------------------------------------------------------------------------------
# For Low-percentage Network : High LR - Low LR
#for prune_rate in 50
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model ConvNet --Train_Batch 64 --Dataset 10 --T_Epoch 10 --LR_step 80 140 160 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9
#done

# For Low-percentage Network : High LR - Low LR
#for prune_rate in 89.5 91 92.5 94 95.5 97
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --Dataset 10 --T_Epoch 150 --LR_step 60 120 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

# For Low-percentage Network : High LR - Low LR
#for prune_rate in 90 92.5
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --Dataset 10 --T_Epoch 160 --LR_step 80 120 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

#for prune_rate in 95
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --Dataset 10 --T_Epoch 200 --LR_step 120 160 --P_rate $prune_rate --lr 0.05 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

# For Coarse-grain only Golden Model
#for prune_rate in 95.38 #91.45
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CO --Train_Batch 64 --Dataset 10 --T_Epoch 200 --LR_step 100 160 --P_rate $prune_rate --lr 0.05 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

#for prune_rate in 75
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --Dataset 10 --T_Epoch 200 --LR_step 80 140 170 190 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

# For High-percentage Network : Low LR - Super Low LR
#for prune_rate in 97
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11 --Train_Batch 64 --Dataset 10 --T_Epoch 200 --LR_step 100 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

#for prune_rate in 10 20 30 40 50 
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 80 140 170 190 --P_rate $prune_rate --lr 0.02 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9
#done

#for prune_rate in 0 10 20 30 40 50 75 78.125 81.25 82.8125 84.375 87.5 93.75 94.53125 95.3125
#for prune_rate in 25 50 75 78.125 81.25 82.8125 
#for prune_rate in 84.375
#for prune_rate in 94.53125 
#for prune_rate in 75 87.5 93.75 94.53125 95.3125
#for prune_rate in 93.75 94.53125 95.3125
#for prune_rate in 81.25 78.125 84.375
#for prune_rate in 82.8125
#for prune_rate in 3.05 5.02 12.8
#for prune_rate in 6.92 8.74 15.96
#for prune_rate in 6.92
#for prune_rate in 8.74 
#for prune_rate in 15.96
#for prune_rate in 0
#for prune_rate in 65.9
#for prune_rate in 37.98
#for prune_rate in 76.74
#for prune_rate in 84.375
#do
	#python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 80 140 170 190 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 80 140 170 190 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9 --gamma 0.5
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 80 140 170 190 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9 --gamma 0.5
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 10 30 60 120 160 180 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9 --gamma 0.5
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --T_Epoch 300 --LR_step 30 60 90 120 150 180 210 240 270 --P_rate $prune_rate --lr 0.05 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.1 --gamma 0.5
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 32 --Dataset 10 --T_Epoch 240 --LR_step 40 80 120 160 200 --P_rate $prune_rate --lr 0.05 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.1
#done

for prune_rate in 30
do
	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 64 --Dataset 10 --T_Epoch 240 --LR_step 40 80 120 160 200 --P_rate $prune_rate --lr 0.0025 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.9
done


#for prune_rate in 60 65 70 75 80 85 90 95 99 
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG7 --Train_Batch 32 --Dataset 10 --T_Epoch 240 --LR_step 40 80 120 160 200 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis f --momentum 0.1
#done

#for prune_rate in 37.98
#	python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --T_Epoch 200 --LR_step 10 30 60 120 160 180 --P_rate $prune_rate --lr 0.1 --Method kc --with_fc t --Dev $dev_num --Vis 0 --momentum 0.9 --gamma 0.5
