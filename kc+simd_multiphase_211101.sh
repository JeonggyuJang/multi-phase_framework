dev_num="3"

#####################################CIFAR100: VGG11 Start from 84.375 #5 phase (from 84 to 0)
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 84 --dP_rate 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd kc --gamma 0.1 --momentum 0

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 84-65 --P_rate 100  --Model VGG11_CIFAR_BIG --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 50 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 480 --LR_step 80 160 240 320 400 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 84-65-53-50 --dP_rate 100 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 84-65-53-50-100 --dP_rate 0  --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd kc --momentum 0
#python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 84-65-53-50-100-0 --P_rate 100  --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
# 63.95
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 84-65-53-50-100-0-65 --dP_rate 50 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3
# 64.59
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 84-65-53-50-100-0-65-50 --dP_rate 100 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3



#hyper parameter fine tune
#echo python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 30 60 90 120 150 180 210 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 84-65-53-50-100-0-65 --dP_rate 50 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3 --gamma 0.5
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 30 60 90 120 150 180 210 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 84-65-53-50-100-0-65 --dP_rate 50 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3 --gamma 0.5
#echo python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 84-65-53-50-100-0-65-50 --dP_rate 100 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 300 --LR_step 60 120 180 240 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 84-65-53-50-100-0-65-50 --dP_rate 100 --init_from 0 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.3




#####################CIFAR100: VGG11 start form 84.375 #5 phase (from 84 to 8, simd 75%)

#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 84 --dP_rate 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd kc --gamma 0.1 --momentum 0

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 84-65 --P_rate 100  --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0


#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5
## Count3 fine tuning...
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0
echo python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 128 --T_Epoch 300 --LR_step 50 100 150 200 250 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.1
python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 128 --T_Epoch 300 --LR_step 50 100 150 200 250 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.1
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.1
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 360 --LR_step 60 120 180 240 300  --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 100 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 10 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 84-65-53 --dP_rate 25 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5

python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 84-65-53-25 --dP_rate 100 --init_from 65 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5

python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 84-65-53-25-100 --dP_rate 8  --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd kc --momentum 0
python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 32 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 84-65-53-25-100-8 --P_rate 100  --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0 

python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 240 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 84-65-53-25-100-8-62 --dP_rate 25 --init_from 8 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5
python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 32 --T_Epoch 340 --LR_step 40 80 120 160 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 84-65-53-25-100-8-62-25 --dP_rate 100 --init_from 8 --Model VGG11_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.1 --momentum 0.5




#####################################CIFAR100: VGG11 Start from 84.375 #5 phase
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 83  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-83 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-83-64 --dP_rate 50 --init_from 83  --Model VGG7 --Vis f --kc_or_simd simd
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-83-64-50 --dP_rate 100 --init_from 83 --Model VGG7 --Vis f --kc_or_simd simd 

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-83-64-50-100 --dP_rate 0  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-83-64-50-100-0 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-83-64-50-100-0-82 --dP_rate 50 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 94-83-64-50-100-0-82-50 --dP_rate 100 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9


#####################################CIFAR10: VGG7 Start from 83.1124% #5 phase
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-88-83 --dP_rate 0  --Model VGG7 --Vis f --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-88-83-0 --P_rate 100  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-88-83-0-82 --dP_rate 25 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-88-83-0-82-25 --dP_rate 33.3333 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-88-83-0-82-25-33 --dP_rate 100 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

#####################################CIFAR10: VGG7 Start from 83.1124% #5 phase
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-88-83 --dP_rate 0  --Model VGG7 --Vis f --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-88-83-0 --P_rate 100  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-88-83-0-82 --dP_rate 50 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-88-83-0-82-50 --dP_rate 100 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

#####################################CIFAR10: VGG7 Start from 94.53125% #7 phase, IEEE_ACCESS
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 83  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-83 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-83-64 --dP_rate 25 --init_from 83  --Model VGG7 --Vis f --kc_or_simd simd
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-83-64-25 --dP_rate 33.3335 --init_from 83 --Model VGG7 --Vis f --kc_or_simd simd 
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-83-64-25-33 --dP_rate 100 --init_from 83 --Model VGG7 --Vis f --kc_or_simd simd 

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-83-64-25-33-100 --dP_rate 0  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-83-64-25-33-100-0 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 94-83-64-25-33-100-0-82 --dP_rate 25 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 9 --Mask 94-83-64-25-33-100-0-82-25 --dP_rate 33.3333 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 10 --Mask 94-83-64-25-33-100-0-82-25-33 --dP_rate 100 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

#####################################CIFAR10: VGG7 Start from 94.53125% #5 phase, IEEE_ACCESS
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 83  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-83 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-83-64 --dP_rate 50 --init_from 83  --Model VGG7 --Vis f --kc_or_simd simd
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-83-64-50 --dP_rate 100 --init_from 83 --Model VGG7 --Vis f --kc_or_simd simd 

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-83-64-50-100 --dP_rate 0  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-83-64-50-100-0 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-83-64-50-100-0-82 --dP_rate 50 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 94-83-64-50-100-0-82-50 --dP_rate 100 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

#####################################CIFAR100: VGG16L_CIFAR_BIG start from 82.5% #5phase

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 82 --dP_rate 58  --Model VGG16L_CIFAR_BIG --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 82-58 --P_rate 100  --Model VGG16L_CIFAR_BIG --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 82-58-57 --dP_rate 39.2755 --init_from 58  --Model VGG16L_CIFAR_BIG --kc_or_simd simd --gamma 0.2
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 82-58-57-39 --dP_rate 100 --init_from 58 --Model VGG16L_CIFAR_BIG --Vis f --kc_or_simd simd --gamma 0.2

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 82-58-57-39-100 --dP_rate 0  --Model VGG16L_CIFAR_BIG --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 100 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 82-58-57-39-100-0 --P_rate 100  --Model VGG16L_CIFAR_BIG --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 100 --LR_step 80 90 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 82-58-57-39-100-0-57 --dP_rate 39.2755 --init_from 0  --Model VGG16L_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.9 --gamma 0.2
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 100 --Train_Batch 64 --T_Epoch 120 --LR_step 80 100 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 82-58-57-39-100-0-57-39 --dP_rate 100 --init_from 0 --Model VGG16L_CIFAR_BIG --Vis f --kc_or_simd simd --momentum 0.9 --gamma 0.2

#####################################CIFAR10: VGG7 Start from 95% #7phase, Restore 0.879311
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 77  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-77 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-77-76 --dP_rate 18.6499 --init_from 77  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-77-76-18 --dP_rate 37.7709 --init_from 77 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-77-76-18-37 --dP_rate 100 --init_from 77 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

### KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-77-76-18-37-100 --dP_rate 0  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-77-76-18-37-100-0 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

### SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 94-77-76-18-37-100-0-76 --dP_rate 18.6499 --init_from 0  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0.9 
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 9 --Mask 94-77-76-18-37-100-0-76-18  --dP_rate 37.7708 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 10 --Mask 94-77-76-18-37-100-0-76-18-37  --dP_rate 100 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9 

#####################################CIFAR10: VGG7 Start from 95% #5 phase, Restore 1.514866
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 94 --dP_rate 77  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 94-77 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 94-77-76 --dP_rate 32.1054 --init_from 77  --Model VGG7 --Vis f --kc_or_simd simd
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 94-77-76-32 --dP_rate 100 --init_from 77 --Model VGG7 --Vis f --kc_or_simd simd 

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 94-77-76-32-100 --dP_rate 0  --Model VGG7 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 94-77-76-32-100-0 --P_rate 100  --Model VGG7 --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 94-77-76-32-100-0-76 --dP_rate 32.1054 --init_from 0  --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 80 140 160 --lr 0.01 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 94-77-76-32-100-0-76-32 --dP_rate 100 --init_from 0 --Model VGG7 --Vis f --kc_or_simd simd --momentum 0.9

#####################################CIFAR10: VGG11 Start from 97.5% #5 phase, Restore 1.514866
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 1 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 84  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 1 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 97-84 --P_rate 100  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 97-84-83 --dP_rate 28.4506 --init_from 84  --Model VGG11 --Vis f --kc_or_simd simd --Maskgen_dePrate 28.4506 100 
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 97-84-83-28 --dP_rate 100 --init_from 84 --Model VGG11 --Vis f --kc_or_simd simd 

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 97-84-83-28-100 --dP_rate 0  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 97-84-83-28-100-0 --P_rate 100  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0 

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 120 --LR_step 60 100 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 97-84-83-28-100-0-84 --dP_rate 28.4506 --init_from 0  --Model VGG11 --Vis t --kc_or_simd simd --Maskgen_dePrate 28.4506 100 --momentum 0.9 --MaskTrain t
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 120 --LR_step 60 100 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 97-84-83-28-100-0-84-28 --dP_rate 100 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 --MaskTrain t


#####################################CIFAR10: VGG11 Start from 97.5% #7phase, Restore 0.879311
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 97 --dP_rate 84  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 늘린영역 다시 없애기.
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 97-84 --P_rate 100  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 97-84-83 --dP_rate 15.95084 --init_from 84  --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 --Maskgen_dePrate 15.94084 45.4489 100
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 97-84-83-15 --dP_rate 45.4489 --init_from 84 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 97-84-83-15-45 --dP_rate 100 --init_from 84 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9

### KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 97-84-83-15-45-0 --dP_rate 0  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 0 --LR_step 10 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 97-84-83-15-45-0-0 --P_rate 100  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0
##
### SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 220 --LR_step 120 180 200 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 97-84-83-70-45-0-0-84 --dP_rate 15.95084 --init_from 0  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0.9 --Maskgen_dePrate 15.94084 45.4489 100
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 170 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 9 --Mask 97-84-83-70-45-0-0-84-70  --dP_rate 45.4489 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 170 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 10 --Mask 97-84-83-70-45-0-0-84-70-45  --dP_rate 100 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 

######################################CIFAR10: VGG11 Start from 98%
# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 160 --LR_step 80 120 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 1 --Mask 98 --dP_rate 87  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9

#SIMD Pruning : 1/7
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 80 120 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 2 --Mask 98-87 --P_rate 85.714 --Model VGG11 --Vis t --kc_or_simd simd --momentum 0

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 3 --Mask 98-87-74 --dP_rate 33.333 --init_from 87  --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 180 --LR_step 100 140 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 4 --Mask 98-87-74-49 --dP_rate 100 --init_from 87 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9

# KC Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 80 120 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 5 --Mask 98-87-74-49-0 --dP_rate 0  --Model VGG11 --Vis t --kc_or_simd kc --momentum 0.9

# SIMD Pruning : momentum 0이 더좋은듯..
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 120 --lr 0.05 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 98-87-74-49-0-0 --P_rate 85.714  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0
#python combined_network_framework.py --P 1 --R 1 --Dataset 10 --Train_Batch 64 --T_Epoch 1 --LR_step 10 --lr 0.05 --Method kc+simd --with_fc t --Dev $dev_num --M_count 6 --Mask 98-87-74-49-0-0 --P_rate 100  --Model VGG11 --Vis t --kc_or_simd simd --momentum 0 #KYUU

# SIMD Multi-phase
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 60 120 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 98-87-74-49-0-0-87  --dP_rate 14.2857 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 #KYUU
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 60 120 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 98-87-74-49-0-0-87-75  --dP_rate 33.3333 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 #KYUU
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 200 --LR_step 60 120 160 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 9 --Mask 98-87-74-49-0-0-87-75-50 --dP_rate 100 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9 #KYUU

#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 7 --Mask 98-87-74-49-0-0-75  --dP_rate 33.333 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9
#python combined_network_framework.py --P 2 --R 1 --M 1 --Dataset 10 --Train_Batch 64 --T_Epoch 140 --LR_step 60 100 --lr 0.1 --Method kc+simd --with_fc t --Dev $dev_num --M_count 8 --Mask 98-87-74-49-0-0-75-50 --dP_rate 100 --init_from 0 --Model VGG11 --Vis f --kc_or_simd simd --momentum 0.9


