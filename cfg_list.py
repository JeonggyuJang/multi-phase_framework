import gzip
import pickle
import argparse
import pprint

def readPickle(fileDir):
    with gzip.open(fileDir, 'rb') as f:
        data = pickle.load(f)
    return data

def writePickle(data, fileDir):
    with gzip.open(fileDir, 'wb') as f:
        pickle.dump(data, f)

def main(filename):
    maskDir = './masks/' + filename
    data = readPickle(maskDir)
    pprint.pprint(data)
    data['cfg_list'] += [(10, data['cfg_list'].pop())] # For CIFAR-10
    pprint.pprint(data)
    check = input('Write this changes to {} Y/N?'.format(filename))
    if check == 'Y':
        writePickle(data,maskDir)
    else:
        print("Wrong input!")
        raise ValueError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname',
            type = str,
            default = None,
            help = 'Maskfile location'
            )
    args = parser.parse_args()
    main(args.fname)
