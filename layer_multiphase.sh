dev_num="0"

#VGG7
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Layer-Multiphase : Fine-grain only from 97.5% 3 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 18.2744 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --with_fc t --Dev $dev_num --M_count 2 --Mask 94-80  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

# Layer-Multiphase : Fine-grain only from 97.5% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 5.8671 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 2 --Mask 94-90  --dP_rate 13.1806 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer  --Dev $dev_num --M_count 3 --Mask 94-90-80  --dP_rate 32.1054 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 4 --Mask 94-90-80-58  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

# Layer_pruning from 95% : 7 Phases

#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 3.4082 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 2 --Mask 94-92  --dP_rate 5.8132 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer  --Dev $dev_num --M_count 3 --Mask 94-92-88  --dP_rate 10.1687 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 4 --Mask 94-92-88-80  --dP_rate 18.6499 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 5 --Mask 94-92-88-80-68  --dP_rate 37.7708 --Model VGG7 --Vis f --momentum 0.9
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 6 --Mask 94-92-88-80-68-46  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9


#VGG11
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Layer-Multiphase : Fine-grain only from 97.5% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 170 --LR_step 80 140 160 --lr 0.01 --Method layer --Dev $dev_num --M_count 1 --Mask 97  --dP_rate 3.8843 --Model VGG11 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 80 140 160 --lr 0.01 --Method layer --with_fc t --Dev $dev_num --M_count 2 --Mask 97-94  --dP_rate 10.1632 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer --with_fc t --Dev $dev_num --M_count 3 --Mask 97-94-87  --dP_rate 28.4506 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer --with_fc t --Dev $dev_num --M_count 4 --Mask 97-94-87-68  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9

# Layer_pruning from 97.5% : 7 Phases
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer  --Dev $dev_num --M_count 1 --Mask 97  --dP_rate 2.1777 --Model VGG11 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer  --Dev $dev_num --M_count 2 --Mask 97-95  --dP_rate 4.1169 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer  --Dev $dev_num --M_count 3 --Mask 97-95-93  --dP_rate 7.9404 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.01 --Method layer  --Dev $dev_num --M_count 4 --Mask 97-95-93-87  --dP_rate 15.9508 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 100 140 --lr 0.01 --Method layer  --Dev $dev_num --M_count 5 --Mask 97-95-93-87-77 --dP_rate 35.0962 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 80 120 --lr 0.01 --Method layer  --Dev $dev_num --M_count 6 --Mask 97-95-93-87-77-58  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9
