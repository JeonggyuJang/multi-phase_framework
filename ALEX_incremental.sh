dev_num="2"

#for prune_rate in 75
#do
#	python multi_phase_network_framework.py --P 1 --R 1 --Model ALEXnet --Train_Batch 64 --Dataset 10 --T_Epoch 220 --LR_step 120 180 200 --P_rate $prune_rate --lr 0.01 --Method kc --with_fc t --Dev $dev_num --Vis 1 --momentum 0.9
#done

## FOR Multiphase : KC : Overlapping Training

#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 220 --LR_step 60 120 160 180 --lr 0.05 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 93 --dP_rate 75  --Model ALEXnet --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 220 --LR_step 60 120 160 180 --lr 0.05 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 93-74 --dP_rate 43  --Model ALEXnet --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 220 --LR_step 60 120 160 180 --lr 0.05 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 93-74-43 --dP_rate 0  --Model ALEXnet --Vis t --momentum 0.9

## FOR Multiphase : KC : Mask Training
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 93 --dP_rate 75  --Model ALEXnet --Vis t --momentum 0.9 --MaskTrain t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 93-74 --dP_rate 43  --Model ALEXnet --Vis t --momentum 0.9 --MaskTrain t
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 93-74-43 --dP_rate 0  --Model ALEXnet --Vis t --momentum 0.9 --MaskTrain t
