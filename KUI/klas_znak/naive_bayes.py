import argparse
import os
from PIL import Image
import numpy as np


NUM_SHADES = 16
PNG_SIZE_BUFF = 28

def setup_arg_parser():
    """Setting up command line arguments"""
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def find_dsv(directory_files):
    """Find .dsv file in directory"""
    for index in range(len(directory_files)):
        if directory_files[index].endswith('.dsv'):
            return index
    return -1

def read_dvs_file(filepath):
    """Read .dsv file and return dictionary of contents"""
    data_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':') 
            data_dict[parts[0]] = parts[1]
    return data_dict

def make_train_data_dict(filenames, train_path):
    """Create a dictionary with image file names as keys and their pixel data as values"""
    image_data_dict = {}
    for image in filenames:
        image_data_dict[image] = np.array(Image.open(f"{train_path}/{image}")).astype(int).flatten() // NUM_SHADES
    return image_data_dict

def write_to_dvs(classified_data, output_path):
    """Write classified data to .dsv file"""
    with open(output_path, 'w') as file:
        for image in classified_data.keys():
            file.write(f"{image}:{classified_data[image]}\n")

def analysis(train_data_dict, train_data_dsv):
    """Analyze the data and calculate the probabilities for each class"""
    image_num = len(list(train_data_dsv.values()))
    class_count = {}
    class_by_pixel = {}
    
    for image in train_data_dsv.keys():
        if train_data_dsv[image] not in class_count:
            class_count[train_data_dsv[image]] = 0
            class_by_pixel[train_data_dsv[image]] = np.zeros((PNG_SIZE_BUFF * PNG_SIZE_BUFF, NUM_SHADES))

        class_count[train_data_dsv[image]] +=1
        for i in range(len((train_data_dict[image]))):
            class_by_pixel[train_data_dsv[image]][i][(train_data_dict[image])[i]] +=1

    prob_count = {}
    prob_by_pixel = {}

    for i in class_count.keys():
        prob_count[i] = class_count[i] / image_num
        prob_by_pixel[i] = (class_by_pixel[i]+1) / (class_count[i] + NUM_SHADES*1)

    return prob_count, prob_by_pixel
            
def classification (prob_count, prob_by_pixel, test_data):
    """Naive Bayes classification"""
    classified_data={}
    for image in test_data.keys():
        max_prob = float("-inf")
        max_class = ''
        for i in prob_count.keys():
            prob = np.log(prob_count[i])
            for index in range(len((test_data[image]))):
                prob+=np.log(prob_by_pixel[i][index][(test_data[image])[index]])
            if prob > max_prob:
                max_prob = prob
                max_class = i
        classified_data[image] = max_class
    return classified_data

def main():
    """Main function where all the procedures are called"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    print("Running Naive Bayes classifier")
    
    train_data = os.listdir(args.train_path)
    test_data = os.listdir(args.test_path)
    dsv_index = find_dsv(train_data)
    train_data_dsv = read_dvs_file(f"{args.train_path}/{train_data[dsv_index]}")

    if dsv_index > 0:
        train_data.remove(train_data[dsv_index])

    filenames = train_data_dsv.keys()
    train_data_dict = make_train_data_dict(filenames ,args.train_path)
    test_data_dict = make_train_data_dict(test_data, args.test_path)

    prob_count, prob_by_pixel = analysis(train_data_dict, train_data_dsv)
    classified_data = classification (prob_count, prob_by_pixel, test_data_dict)
    write_to_dvs(classified_data, args.o)

if __name__ == "__main__":
    main()
