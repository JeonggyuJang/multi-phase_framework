import pickle

#with open('VGG11_CIFAR_BIG_kc_kc_info_10%_pruned.pickle','rb') as f:
with open('VGG7_kc+simd_kc_info_94-88-83-0-82-25-33-100_multi_phased_7#.pickle','rb') as f:
    data = pickle.load(f)

print("<data>\n",data)
