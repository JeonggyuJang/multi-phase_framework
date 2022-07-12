dev_num="3"
model_name="VGG7"
#model_name="VGG16L_CIFAR_BIG"
#model_name="VGG11_CIFAR_BIG"
#prune_method="gpu_micro"
prune_method="kc"
#prune_method="kc+simd"
#prune_method="simd"
#process="multi_phased"
process="pruned"
#multi_count="3"
#dataset="100"
dataset="10"
with_fc="t"




###test_filename=$prune_method"_"$process"_84-65%_1#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53%_2#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50%_3#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100%_4#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53-50-100-0%_5#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65%_6#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65-50%_7#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65-50 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65-50-100%_8#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65-50-100 --Method $prune_method --with_fc $with_fc 


###test_filename=$prune_method"_"$process"_84-65%_1#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53%_2#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-25%_3#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-25 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100%_4#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53-50-100-0%_5#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0 --Method $prune_method --with_fc $with_fc 
###test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65%_6#"
###python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65-50%_7#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65-50 --Method $prune_method --with_fc $with_fc 
#test_filename=$prune_method"_"$process"_84-65-53-50-100-0-65-50-100%_8#"
#python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask 84-65-53-50-100-0-65-50-100 --Method $prune_method --with_fc $with_fc 

# Test for KC pruning network : For Multi-phased Network
#for prune_rate in 94-89-77-52
#do
#    test_filename=$prune_method"_"$process"_"$prune_rate"%_"$multi_count"#"
#    python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask $prune_rate --Method $prune_method --with_fc $with_fc --M_count $multi_count
#done
python multi_phase_network_framework.py --Model $model_name --Test "pretrained"  --Dev $dev_num --Dataset $dataset --Method $prune_method --with_fc $with_fc --Mask 0 

#for prune_rate in 98 94 89 84 79 75 69 64 59 55 50 39 30 19 10 
#for prune_rate in 7 8 16
#for prune_rate in 84
#do
#    echo Ratio:$prune_rate%
#    test_filename=$prune_method"_"$process"_"$prune_rate"%"
#	python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask $prune_rate --Method $prune_method --with_fc $with_fc 
#done

# Test for KC pruning network
#for prune_rate in 50
#do
#    test_filename=$prune_method"_"$process"_"$prune_rate"%"
#	python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Method $prune_method --with_fc $with_fc --Mask $prune_rate
#done

#for prune_rate in 55
#do
#    test_filename=$prune_method"_"$process"_"$prune_rate"%"
#    python multi_phase_network_framework.py --Model $model_name --Test $test_filename --Dev $dev_num --Dataset $dataset --Mask $prune_rate --Method $prune_method --with_fc $with_fc 
#done

