#for prune_rate in 20 30 40 50 75 81 82 87
#for prune_rate in 3 5 12
#for prune_rate in 7 8 16
#for prune_rate in 7
#for prune_rate in 76 65 38 8 
#do
#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 64 --Dataset 100 --P_rate $prune_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --dP_rate 0 --M_count 1 --Mask $prune_rate
#    tar_name='VGG11_kc_'$prune_rate'_pruned.tar'
#    tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#    scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp
#done


#for prune_rate in 10 19 30 39 50 55 59 64 69 75 79 84 89 94 98
#for prune_rate in 10 19 30
#do
#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG7 --Train_Batch 32 --Dataset 10 --P_rate $prune_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --dP_rate 0 --M_count 1 --Mask $prune_rate
#    tar_name='VGG7_kc_'$prune_rate'_pruned.tar'
#    tar -cvf ./metadata/$tar_name ./metadata/VGG7conv*.csv
#    scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp
#done

#    python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --P_rate $prune_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --dP_rate 0 --M_count 1 --Mask $prune_rate
#    tar_name='VGG11_kc_'$prune_rate'_pruned.tar'
#    tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#    scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp



#prune_rate='84-65-53'
#dP_rate='25'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 65 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 3 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-65-53-25'
#dP_rate='100'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 65 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 4 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-65-53-25-100-8-62'
#dP_rate='25'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 8 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 7 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-65-53-25-100-8-62-25'
#dP_rate='100'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 8 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 8 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


#prune_rate='84-65-53-50'
#dP_rate='100'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 65 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 4 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-65-53-50-100-0-65'
#dP_rate='50'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 0 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 7 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-65-53-50-100-0-65-50'
#dP_rate='100'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --init_from 0 --Method kc+simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 8 --Mask $prune_rate --kc_or_simd simd
#tar_name='VGG11_kc+simd_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

############
#for coarse 5p #error..
#prune_rate='84'
#dP_rate='76'
#python combined_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 1 --Mask $prune_rate --kc_or_simd kc
#tar_name='VGG11_kc_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


#prune_rate='84-76'
#dP_rate='65'
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 2 --Mask $prune_rate 
#tar_name='VGG11_kc_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


#prune_rate='84-76-65'
#dP_rate='38'
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 3 --Mask $prune_rate
#tar_name='VGG11_kc_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


#prune_rate='84-76-65-38'
#dP_rate='8'
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 4 --Mask $prune_rate
#tar_name='VGG11_kc_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#prune_rate='84-76-65-38-8'
#dP_rate='8'
#python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method kc --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 4 --Mask $prune_rate
#tar_name='VGG11_kc_'$prune_rate'-'$dP_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

#####
#for fine 5p
#prune_rate='84.375'
#python multi_phase_network_framework.py --P 1 --R 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --Method simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 0
#tar_name='VGG11_simd_'$prune_rate'_pruned.tar'
#tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
#scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp

prune_rate='84'
dP_rate='9'
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 1 --Mask $prune_rate --kc_or_simd simd
tar_name='VGG11_simd_'$prune_rate'-'$dP_rate'_pruned.tar'
tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


prune_rate='84-76'
dP_rate='14'
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 2 --Mask $prune_rate --kc_or_simd simd
tar_name='VGG11_simd_'$prune_rate'-'$dP_rate'_pruned.tar'
tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


prune_rate='84-76-65'
dP_rate='42'
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 3 --Mask $prune_rate --kc_or_simd simd
tar_name='VGG11_simd_'$prune_rate'-'$dP_rate'_pruned.tar'
tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


prune_rate='84-76-65-38'
dP_rate='78'
python multi_phase_network_framework.py --P 2 --R 1 --M 1 --Model VGG11_CIFAR_BIG --Train_Batch 32 --Dataset 100 --dP_rate $dP_rate --Method simd --with_fc t --Dev 0 --Vis 0 --Meta 1 --M_count 4 --Mask $prune_rate  --kc_or_simd simd
tar_name='VGG11_simd_'$prune_rate'-'$dP_rate'_pruned.tar'
tar -cvf ./metadata/$tar_name ./metadata/VGG11_CIFAR_BIGconv*.csv
scp -P 9000 ./metadata/$tar_name moco@202.30.11.129:/home/moco/temp


