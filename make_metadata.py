import argparse
import pdb
from pruning.utils import save_metadata_scattered, data_load_pickle, save_metadata

def main(Mask_filename, NN_name):
    if Mask_filename == None:
        raise ValueError
    else:
        mask_loc = './masks/'+Mask_filename
    masks = data_load_pickle(mask_loc)
    #pdb.set_trace()

    save_metadata(masks, NN_name, 2 , RBS_MAX = 3008, is_encode = True, is_fc_inc = True)
    #save_metadata_scattered(masks, NN_name, 2 , RBS_MAX = 3008, is_encode = True, is_fc_inc = True)
    #save_metadata_scattered(masks, NN_name, 2 , RBS_MAX = 3008, is_encode = True, is_fc_inc = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'For saving Metadata')
    parser.add_argument('--Mask_filename',
            type=str,
            default=None
    )
    parser.add_argument('--Model',
            type=str,
            default=None
    )
    args = parser.parse_args()
    main(args.Mask_filename, args.Model)

