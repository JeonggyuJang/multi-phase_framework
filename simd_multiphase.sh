dev_num="3"
# SIMD-Multiphase : VGG11_BIG Fine-grain only from 84% 3 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 250 --LR_step 100 150 200 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 84  --dP_rate 22 --Model VGG11_CIFAR_BIG --Vis f --momentum 0.9 --gamma 0.1
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 250 --LR_step 100 150 200 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 84-65  --dP_rate 87 --Model VGG11_CIFAR_BIG --Vis f --momentum 0.9 --gamma 0.1


# SIMD-Multiphase : VGG11_BIG Fine-grain only from 84% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 84  --dP_rate 9 --Model VGG11_CIFAR_BIG --Vis f --momentum 0
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 84-76  --dP_rate 14 --Model VGG11_CIFAR_BIG --Vis f --momentum 0
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-76-65  --dP_rate 42 --Model VGG11_CIFAR_BIG --Vis f --momentum 0
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 84-76-65-38  --dP_rate 78 --Model VGG11_CIFAR_BIG --Vis f --momentum 0.2
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 200 --LR_step 40 80 120 160 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 84-76-65-38  --dP_rate 78 --Model VGG11_CIFAR_BIG --Vis f --momentum 0.2
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 40 80 120 160 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 84-76-65-38  --dP_rate 78 --Model VGG11_CIFAR_BIG --Vis f --momentum 0.2 --Meta 1




# SIMD-MultiPhase : 살리는 영역 고찰, from 95%
# 94% ~ 90%
#for dePrune_rate in 1.0526 2.10526 3.15789 4.210526 5.263158
#do 
#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 140 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate $dePrune_rate --Model VGG11 --Vis f --momentum 0.9
#done
# 89% ~ 85%
#for dePrune_rate in 6.315789 7.368421 8.421053 9.473684 10.526316
#do 
#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 140 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate $dePrune_rate --Model VGG11 --Vis f --momentum 0.9
#done
# 80% 75% 70% 60%
#for dePrune_rate in 15.789474 21.052632 26.315789 36.842105
#do 
#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 140 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate $dePrune_rate --Model VGG11 --Vis f --momentum 0.9
#done
#------------------------------------------------------------------------------------------------------------------------------------------------------
# SIMD-Multiphase : VGG16L_CIFAR_BIG , 5Phase
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 82  --dP_rate 11.5841 --Model VGG16L_CIFAR_BIG --Vis f --momentum 0.9 --Dataset 100 --Weight_decay 0.0005 --gamma 0.2
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 82-72  --dP_rate 20.2570 --Model VGG16L_CIFAR_BIG --Vis f --momentum 0.9 --Dataset 100 --Weight_decay 0.0005 --gamma 0.2
#
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 82-72-58  --dP_rate 38.2756 --Model VGG16L_CIFAR_BIG --Vis f --momentum 0.9 --Dataset 100 --Weight_decay 0.0005 --gamma 0.2
#
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 200 --LR_step 80 160 180 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 82-72-58-35  --dP_rate 100 --Model VGG16L_CIFAR_BIG --Vis f --momentum 0.9 --Dataset 100 --Weight_decay 0.0005 --gamma 0.2


# SIMD-Multiphase : Fine-grain only from 94.53% DATE
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 12.0794 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-82  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

# SIMD-Multiphase : Fine-grain only from 95% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 6.0397 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-88  --dP_rate 6.4279 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-88-82  --dP_rate 50 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-88-82-41  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

# SIMD-Multiphase : VGG7 Fine-grain only from 95% : 7Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 3.0199 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-91  --dP_rate 3.1139 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-91-88  --dP_rate 6.4279 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-91-88-82  --dP_rate 25 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-91-88-82-62  --dP_rate 33.3333 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-91-88-82-62-41  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

#------------------------------------------------------------------------------------------------------------------------------------------------------
# SIMD-Multiphase : Fine-grain only from 95% 3 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 74  --dP_rate 0 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-77  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

# SIMD-Multiphase : Fine-grain only from 95% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 5.8671 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-89  --dP_rate 13.1806 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-89-77  --dP_rate 32.1054 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-89-77-52  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9
#
# SIMD-Multiphase : VGG7 Fine-grain only from 95% : 7Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94  --dP_rate 3.4082 --Model VGG7 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-91  --dP_rate 5.8132 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-91-86  --dP_rate 10.1687 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-91-86-77  --dP_rate 18.6499 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-91-86-77-62  --dP_rate 37.7708 --Model VGG7 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-91-86-77-62-39  --dP_rate 100 --Model VGG7 --Vis f --momentum 0.9

 #SIMD-Multiphase : Fine-grain only from 97.5% 3 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 97  --dP_rate 13.6527 --Model VGG11 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 97-93  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9

# SIMD-Multiphase : Fine-grain only from 97.5% 5 Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 97  --dP_rate 3.8843 --Model VGG11 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 97-93  --dP_rate 10.1632 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 97-93-84  --dP_rate 28.4506 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 60 120 140 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 97-93-84-60  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9

# SIMD-Multiphase : Fine-grain only from 97.5% : 7Phase!!!!
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 97  --dP_rate 2.1777 --Model VGG11 --Vis f --momentum 0.9 
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 97-95  --dP_rate 4.1169 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 97-95-91  --dP_rate 7.9404 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 100 160 180 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 97-95-91-84  --dP_rate 15.9508 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 170 --LR_step 100 140 160 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 5 --Mask 97-95-91-84-70  --dP_rate 35.0962 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 100 --LR_step 40 80 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 6 --Mask 97-95-91-84-70-45  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9
#
# SIMD-Multiphase : Fine-grain only from 98.5% , 살리는 비율 1
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 140 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 98  --dP_rate 1.5228 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 98-96  --dP_rate 3.0927 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 3 --Mask 98-96-93  --dP_rate 6.383 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 4 --Mask 98-96-93-87  --dP_rate 13.6363 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 5 --Mask 98-96-93-87-75  --dP_rate 31.5789 --Model VGG11 --Vis f --momentum 0.9
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 --lr 0.1 --Method simd --with_fc t --Dev $dev_num --M_count 6 --Mask 98-96-93-87-75-51  --dP_rate 100 --Model VGG11 --Vis f --momentum 0.9
#CIFAR100
#python ../multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 300 --LR_step 80 --lr 0.01 --Method simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 50 --Model VGG16_CIFAR_BIG --Vis t
#python ../multi_phase_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 1024 --T_Epoch 500 --LR_step 200 --lr 0.001 --Method simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-47 --dP_rate 100  --Model VGG16_CIFAR_BIG --Vis t
