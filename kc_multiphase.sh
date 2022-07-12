dev_num="1"

#CIFAR10: VGG7- 94 / 3-Phase / ieee_access
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 91  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-91 --dP_rate 88  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-91-88 --dP_rate 83  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 94-91-88-83 --dP_rate 62  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 5 --Mask 94-91-88-83-62 --dP_rate 41  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 6 --Mask 94-91-88-83-62-41 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#CIFAR10: VGG7- 94 / 5-Phase / ieee_access
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 88  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-88 --dP_rate 83  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-88-83 --dP_rate 41  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 94-88-83-41 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#CIFAR10: VGG7- 94 / 3-Phase / ieee_access
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 83  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-83 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#----------------------------------------------------------------------------------------------------------------------------------
# FOR VGG16L (2x FC) / 5-Phase

#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 82 --dP_rate 72  --Model VGG16L_CIFAR_BIG --Vis t --momentum 0.9 --Dataset 100 --gamma 0.2
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 82-72 --dP_rate 58  --Model VGG16L_CIFAR_BIG --Vis t --momentum 0.9 --Dataset 100 --gamma 0.2
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 82-72-58 --dP_rate 35  --Model VGG16L_CIFAR_BIG --Vis t --momentum 0.9 --Dataset 100 --gamma 0.2
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90  --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 82-72-58-35 --dP_rate 0  --Model VGG16L_CIFAR_BIG --Vis t --momentum 0.9 --Dataset 100 --gamma 0.2


#----------------------------------------------------------------------------------------------------------------------------------
#CIFAR10: VGG7- 95 / 3-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 77  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-77 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#CIFAR10: VGG7- 95 / 5-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 89  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-89 --dP_rate 77  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-89-77 --dP_rate 52  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160  --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 94-89-77-52 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#CIFAR10: VGG7- 95 / 7-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 91  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-91 --dP_rate 86  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-91-86 --dP_rate 77  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 94-91-86-77 --dP_rate 63  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 5 --Mask 94-91-86-77-63 --dP_rate 39  --Model VGG7 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 140 160 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 6 --Mask 94-91-86-77-63-39 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9

#----------------------------------------------------------------------------------------------------------------------------------
# Initial Training for VGG11
#python multi_phase_network_framework.py --T 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num  --Model VGG11_48 --momentum 0.9
#python multi_phase_network_framework.py --T 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num  --Model VGG11_32 --momentum 0.9
#python multi_phase_network_framework.py --T 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num  --Model VGG11 --momentum 0.9
#python multi_phase_network_framework.py --T 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num  --Model VGG11_16 --momentum 0.9

# FOR Meta_flag Generation
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 62 --dP_rate 0  --Model VGG7 --Vis t --momentum 0.9 --Meta 1

#CIFAR10: VGG11- 97.5 / 3-Phase : KCC2020
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 80 140 170 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 93 --dP_rate 75  --Model VGG11 --Vis t --momentum 0.9 --MaskTrain f
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 80 140 170 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 93-74 --dP_rate 43  --Model VGG11 --Vis t --momentum 0.9 --MaskTrain f
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 80 140 170 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 93-74-43 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9 --MaskTrain f

#CIFAR10: VGG11- 97.5 / 3-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 84  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 110 --LR_step 80 100 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 97-84 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9

#CIFAR10: VGG11- 97.5 / 5-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 93  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 97-93 --dP_rate 84  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 97-93-84 --dP_rate 60  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 110 --LR_step 80 100  --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 97-93-84-60 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9

#CIFAR10: VGG11- 97.5 / 7-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 95  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 97-95 --dP_rate 91  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 97-95-91 --dP_rate 84  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 97-95-91-84 --dP_rate 70  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 5 --Mask 97-95-91-84-70 --dP_rate 46  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 6 --Mask 97-95-91-84-70-46 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9

#CIFAR10: VGG11- 98.5 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 98 --dP_rate 96  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 98-96 --dP_rate 94  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 98-96-94 --dP_rate 87  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 98-96-94-87 --dP_rate 76  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 5 --Mask 98-96-94-87-76 --dP_rate 52  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 6 --Mask 98-96-94-87-76-52 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9

#CIFAR10: VGG11
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 120 --LR_step 40 80 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 87  --Model VGG11 --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 40 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-84 --dP_rate 70  --Model VGG11 --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-84-70 --dP_rate 40  --Model VGG11 --Vis t 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 250 --LR_step 100 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 94-84-70-40 --dP_rate 0  --Model VGG11 --Vis t 

#CIFAR10: VGG11_WO_BN
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 80 --LR_step 50 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 70  --Model VGG11_WO_BN --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 80 --LR_step 50 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-70 --dP_rate 40  --Model VGG11_WO_BN --Vis t 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 80 --LR_step 50 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-70-40 --dP_rate 0  --Model VGG11_WO_BN --Vis t 

#CIFAR100: VGG16
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 150 --LR_step 100 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 50  --Model VGG16_CIFAR_BIG --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 150 --LR_step 100 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-50 --dP_rate 0  --Model VGG16_CIFAR_BIG --Vis t

#CIFAR100: VGG16 : 3_phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 140 --LR_step 80 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 70  --Model VGG16_CIFAR_BIG --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 140 --LR_step 80 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 94-70 --dP_rate 40  --Model VGG16_CIFAR_BIG --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 140 --LR_step 80 --lr 0.01 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 94-70-40 --dP_rate 0  --Model VGG16_CIFAR_BIG --Vis t

#CIFAR100: VGG16L : 3_phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 140 --LR_step 100 120 --lr 0.1 --Method kc --with_fc f --Dev $dev_num --M_count 1 --Mask 90 --dP_rate 53  --Model VGG16L_CIFAR_BIG --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 140 --LR_step 100 120 --lr 0.1 --Method kc --with_fc f --Dev $dev_num --M_count 2 --Mask 90-67 --dP_rate 25  --Model VGG16L_CIFAR_BIG --Vis t
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 150 --LR_step 80 110 130 --lr 0.1 --Method kc --with_fc f --Dev $dev_num --M_count 3 --Mask 90-67-37 --dP_rate 0  --Model VGG16L_CIFAR_BIG --Vis t

#--------------------------------- IEEE_ACCESS
#CIFAR10: VGG11- 97.5 / 3-Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 120 160 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 84  --Model VGG11 --Vis t --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 110 --LR_step 80 100 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 97-84 --dP_rate 0  --Model VGG11 --Vis t --momentum 0.9


#CIFAR100: VGG11_BIG : 3_phase
echo python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 84 --dP_rate 65  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1 --momentum 0.5
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 84 --dP_rate 65  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1 --momentum 0.5


#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 84-65 --dP_rate 8  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1 --momentum 0.5


#CIFAR100: VGG11_BIG : 5_phase

#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 1 --Mask 84 --dP_rate 76  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 2 --Mask 84-76 --dP_rate 65  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 3 --Mask 84-76-65 --dP_rate 38  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc --with_fc t --Dev $dev_num --M_count 4 --Mask 84-76-65-38 --dP_rate 8  --Model VGG11_CIFAR_BIG --Vis f --gamma 0.1


