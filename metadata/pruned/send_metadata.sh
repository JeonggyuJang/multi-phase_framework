#tar_name='./VGG_CIFAR_BIG_81_pruned.tar'
#tar_name='./VGG7_kc_pruned_77%.tar'
#tar_name='./VGG11_kc_multi_phased_95-70-40-0%.tar'
#tar_name='VGG11_kc+simd_multi_phased_94-70-41%_1#.tar'
tar_name='./VGG11_pretrained.tar'
#tar -cvf $tar_name ./VGG11_CIFAR_BIGconv*.csv
tar -cvf $tar_name ./VGG11conv*.csv
#tar -cvf $tar_name ./*.csv
#scp -P 9000 $tar_name moco@202.30.11.129:/moco/temp # JJG AGX
scp -P 9000 $tar_name moco@202.30.11.129:/home/moco/temp
#scp -P 9000 ./VGG11conv*.csv moco@202.30.11.129:/home/moco/nest_workspace/nest_darknet_with_nnpack_project/darknet-nnpack/cfg/encoded_meta_data_dir/vgg11/VGG11_simd_pruned_87%_sc_en_rbs_768 # JJG AGX
#scp -P 3002 $tar_name kyuu@202.30.11.122:/home/kyuu/temp # KYUU AGX
#scp -P 3002 ./VGG11_FCfc*.csv kyuu@202.30.11.122:/home/kyuu/temp # KYUU AGX
