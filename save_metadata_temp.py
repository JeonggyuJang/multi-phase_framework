import numpy as np
original_num_filters = [64, 128, 256, 256, 512, 512, 512, 512]
level_shape_77 = [
        (45, 51, 103, 103, 205, 205, 205, 512),
        (52, 79, 159, 159, 318, 318, 318, 512),
        (56, 96, 192, 192, 385, 385, 385, 512),
        (60, 112, 225, 225, 452, 452, 452, 512),
        (64, 128, 256, 256, 512, 512, 512, 512)
        ]
level_shape_88 = []

def save_metadata(NN_name, RBS_MAX = 256):
    for lv_ind, level_shape in enumerate(level_shape_77):
        # Generate masks
        masks = []
        for ind, num_filter in enumerate(original_num_filters):
            if ind == 0:
                num_channel, width, height = (3, 3, 3)
                num_level_channel, num_level_width, num_level_height = (3, 3, 3)
                filter_size = num_channel * width * height
                level_filter_size = num_level_channel * num_level_width * num_level_height
            else:
                num_channel, width, height = (original_num_filters[ind-1], 3, 3)
                num_level_channel, num_level_width, num_level_height = (level_shape[ind-1], 3, 3)
                filter_size = num_channel * width * height
                level_filter_size = num_level_channel * num_level_width * num_level_height

            mask = np.zeros((num_filter, num_channel * width * height))
            print("Initialize Metadata for" + str(ind) + "th Conv Layer.. : ", mask.shape)
            print("Flip mask for each level.. " + str(ind) + "th Conv Layer.. : ", (num_level_channel, num_level_width, num_level_height))
            mask[:level_shape[ind], :level_filter_size] = 1
            mask = mask.reshape(num_filter, num_channel, width, height)
            masks.append(mask)

        for i, layer in enumerate(masks):
            print("Saving Metadata for" + str(i) + "th Conv Layer..")
            meta_buffer = []
            fname = './temp/' + 'lv' + str(lv_ind) + NN_name + 'conv' + str(i) + '.csv'

            num_filter, channel, width, height = layer.shape
            filter_size = channel * width * height
            temp_layer = layer.reshape(num_filter, filter_size)
            num_filter_divided = num_filter // 4
            print("num_filter_divided_4 and filter size = {}, {}".format(num_filter_divided, filter_size))

            # Set Reduction Block Size
            if filter_size < RBS_MAX:
                reduction_block_size = filter_size
                num_reduction_block = 1
            else:
                reduction_block_size = RBS_MAX
                num_reduction_block = filter_size // reduction_block_size

            print("num_reduction_block = {}".format(num_reduction_block))
            # Extract Metadata information from RBS
            for reduction_block_ind in range(num_reduction_block):
                reduction_block_start = reduction_block_ind * reduction_block_size
                for filter_ind_f in range(0, num_filter, 4):
                    cnt = 0
                    # filter size < 256
                    if filter_size < RBS_MAX:
                        for ind in range(reduction_block_size):
                            if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                                cnt += 1
                            else: break
                    # filter size >= RBS_MAX
                    else:
                        for ind in range(reduction_block_size):
                            if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                                cnt += 1
                            else: break
                    meta_buffer.append(cnt)

            # Extract Metadata information from LeftOver
            if filter_size % reduction_block_size !=0:
                leftover_block_start = (reduction_block_ind+1) * reduction_block_size
                leftover_block_size = filter_size - leftover_block_start
                for filter_ind_f in range(0, num_filter, 4):
                    cnt = 0
                    for ind in range(leftover_block_size):
                        if temp_layer[filter_ind_f][leftover_block_start + ind] == 1:
                            cnt += 1
                        else: break
                    meta_buffer.append(cnt)
            #print(meta_buffer)

            np.savetxt(fname, np.array(meta_buffer).astype(int), delimiter = '\n', fmt='%d')
    return


save_metadata('VGG11', RBS_MAX = 256)
