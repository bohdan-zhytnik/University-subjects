import argparse
import os
from PIL import Image
import numpy as np
from collections import Counter


def setup_arg_parser():
    """Setting up command line arguments"""
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='Path to the training data directory')
    parser.add_argument('test_path', type=str, help='Path to the testing data directory')
    parser.add_argument('-k', type=int, 
                        help='Run k-NN classifier. If k is 0, the code may decide about proper K by itself')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="Path (including the filename) of the output .dsv file with the results")
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
        image_data_dict[image] = np.array(Image.open(f"{train_path}/{image}")).astype(int).flatten()
    return image_data_dict

def get_dist(image1, image2):
    """Calculate Euclidean distance between two images"""
    difference = image1 - image2
    distance = np.sqrt(np.sum(np.square(difference)))
    return distance

def classification(train_data_dict, test_data_dict, train_data_dvs, test_path, k_value):
    """k-NN classification"""
    classified_data = {}
    for image in test_data_dict.keys():
        img_distance_arr = []

        for train_image in train_data_dict.keys():
            distance = get_dist(test_data_dict[image], train_data_dict[train_image])
            img_distance_arr.append(distance)
        
        distances_arr = np.array(img_distance_arr)
        train_img_indices = np.argpartition(distances_arr, k_value, axis=0)[:k_value]

        dvs_values = []
        for i in train_img_indices:
            dvs_values.append(list(train_data_dvs.values())[i])
        counter = Counter(dvs_values)
        most_common_value, _ = counter.most_common(1)[0]
        classified_data[image] = most_common_value
    return classified_data



def write_to_dvs(classified_data, output_path):
    """Write classified data to .dsv file"""
    with open(output_path, 'w') as file:
        for image in classified_data.keys():
            file.write(f"{image}:{classified_data[image]}\n")

def main():
    """Main function to run the script"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    print(f"Running k-NN classifier with k={args.k}")

    train_data = os.listdir(args.train_path)
    test_data = os.listdir(args.test_path)
    dsv_index = find_dsv(train_data)
    
    train_data_dvs = read_dvs_file(f"{args.train_path}/{train_data[dsv_index]}")

    if dsv_index > 0:
        train_data.remove(train_data[dsv_index])

    file_names = train_data_dvs.keys()
    train_data_dict = make_train_data_dict(file_names, args.train_path)
    test_data_dict = make_train_data_dict(test_data, args.test_path)

    classified_data = classification(train_data_dict, test_data_dict, train_data_dvs, args.test_path, args.k)
    write_to_dvs(classified_data, args.o)
        
        
if __name__ == "__main__":
    main()
