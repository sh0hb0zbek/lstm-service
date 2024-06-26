# ------------------------------------------------------------------------
# Implemented code from https://github.com/theAIGuysCode/OIDv4_ToolKit.git
# ------------------------------------------------------------------------

import argparse
from config import cfg
import os
import urllib.request
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import cv2
import time
import sys
import numpy as np
import yaml

ROOT_DIR = os.getcwd()


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes',
                        required=True,
                        nargs='+',
                        metavar='list of classes',
                        help="Sequence of 'strings' of the wanted classes")
    parser.add_argument('--type_csv',
                        required=True,
                        choices=['train', 'test', 'val', 'all'],
                        metavar="'train' or 'val' or 'test' or 'all'",
                        help='From what csv search the images')
    parser.add_argument('--limit',
                        required=False,
                        type=int,
                        default=None,
                        metavar='integer number',
                        help='Optional limit on number of images to download')
    parser.add_argument('--do_print',
                        required=False,
                        action='store_true',
                        help='Optional to print log messages')
    parser.add_argument('--dont_annotate',
                        required=False,
                        action='store_true',
                        help='Optional not to convert annotations for labelling')
    parser.add_argument('--dont_save_yaml',
                        required=False,
                        action='store_true',
                        help='Optional not to save dataset paths to .yaml file')
    return parser.parse_args()


class Downloader:
    def __init__(self, args):
        self.classes = args.classes
        self.csv_dir = None
        self.type_csv = args.type_csv
        self.limit = args.limit
        self.do_print = args.do_print
        self.dont_annotate = args.dont_annotate
        self.dont_save_yaml = args.dont_save_yaml
        self.bc = cfg.BCOLORS
        self.file_class_csv = 'class-descriptions-boxable.csv'
        self.file_list_csv = {
            'train': 'train-annotations-bbox.csv',
            'val': 'validation-annotations-bbox.csv',
            'test': 'test-annotations-bbox.csv'}
    
    def main(self):
        # download images
        self.downloader()
        # convert annotations
        if not self.dont_annotate:
            self.convert_annotations()
        # save dataset path to .yaml file
        if not self.dont_save_yaml:
            self.save_yaml()
    
    def downloader(self):
        self.dataset_dir = os.path.join(ROOT_DIR, cfg.DATASET_DIR)
        self.csv_dir = os.path.join(self.dataset_dir, 'csv_folder')
        CLASSES_CSV = os.path.join(self.csv_dir, self.file_class_csv)
        if self.do_print:
            print(self.bc.OKGREEN + f'Download for {self.type_csv} starts!' + self.bc.ENDC)
        if self.classes[0].endswith('.txt'):
            with open(self.classes[0]) as f:
                self.classes = f.readlines()
                self.classes = [x.strip() for x in self.classes]
        else:
            self.classes = [arg.replace('_', ' ') for arg in self.classes]
        self.mkdirs()
        for classes in self.classes:
            class_name = classes
            if self.do_print:
                print(self.bc.OKGREEN + f'Download {classes}' + self.bc.ENDC)
            self.error_csv(self.file_class_csv)
            df_classes = pd.read_csv(CLASSES_CSV, header=None)
            class_code = df_classes.loc[df_classes[1] == class_name].values[0][0]
            if self.type_csv != 'all':
                name_file = self.file_list_csv.get(self.type_csv)
                df_val = self.TTV(name_file)
                self.download(df_val, self.type_csv, class_name, class_code)
            elif self.type_csv == 'all':
                for type_csv in self.file_list_csv.keys():
                    name_file = self.file_list_csv.get(type_csv)
                    df_val = self.TTV(name_file)
                    self.download(df_val, type_csv, class_name, class_code)
            else:
                print(self.bc.ERROR + 'csv file not specified' + self.bc.ENDC)
                exit(1)
    
    def mkdirs(self):
        '''
        Make the folder structure for the system.
        :return: None
        '''
        
        directory_list = ['train', 'val', 'test']
        folders = ['labels', 'images']
        
        if self.type_csv == 'all':
            for data_type in directory_list:
                for folder_type in folders:
                    folder = os.path.join(self.dataset_dir, data_type, folder_type)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
        else:
            for folder_type in folders:
                folder = os.path.join(self.dataset_dir, self.type_csv, folder_type)
                if not os.path.exists(folder):
                    os.makedirs(folder)
        
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        
    def error_csv(self, file):
        '''
        Check the presence of the required .csv files.
        :param file: .csv file missing
        :return: None
        '''
        try:
            if not os.path.isfile(os.path.join(self.csv_dir, file)):
                folder = str(os.path.basename(file)).split('-')[0]
                if folder != 'class':
                    FILE_URL = str(cfg.OID_URL + folder + '/' + file)
                else:
                    FILE_URL = str(cfg.OID_URL + file)
                FILE_PATH = os.path.join(self.csv_dir, file)
                if self.do_print:
                    urllib.request.urlretrieve(FILE_URL, FILE_PATH, self.reporthook)
                else:
                    urllib.request.urlretrieve(FILE_URL, FILE_PATH)
        except Exception as e:
            exit(1)
    
    def reporthook(self, count, block_size, total_size):
        '''
        Print the progression bar for the .csv file download.
        :param count:
        :param block_size:
        :param total_size:
        :return:
        '''
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / ((1024 * duration) + 1e-5))
        percent = int(count * block_size * 100 / (total_size + 1e-5))
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()
    
    def TTV(self, name_file):
        '''
        Manage error_csv and read the correct .csv file.
        :param name_file: name of the correct .csv file
        :return: None
        '''
        CSV = os.path.join(self.csv_dir, name_file)
        self.error_csv(name_file)
        df_val = pd.read_csv(CSV)
        return df_val
    
    def download(self, df_val_images, folder, class_name, class_code):
        '''
        Manage the download of the images and the label maker.
        :param df_val: DataFrame Values
        :param folder: train, val or test
        :param class_name: self explanatory
        :param class_code: self explanatory
        :return: None
        '''
        if self.do_print:
            if os.name == 'posix':
                rows, columns = os.popen('stty size', 'r').read().split()
            elif os.name == 'nt':
                try:
                    columns, rows = os.get_terminal_size(0)
                except OSError:
                    columns, rows = os.get_terminal_size(1)
            l = int((int(columns) - len(class_name)) / 2)
            print('\n' + self.bc.HEADER + '-'*l + class_name + '-'*l + self.bc.ENDC)
            print(self.bc.INFO+ f'Downloading {folder} images.' + self.bc.ENDC)
        images_list = df_val_images['ImageID'][df_val_images.LabelName == class_code].values
        images_list = set(images_list)
        if self.do_print:
            print(self.bc.INFO + f'Found {len(images_list)} online images for {folder}.' + self.bc.ENDC)
        if self.limit is not None:
            import itertools
            if self.do_print:
                print(self.bc.INFO + f'Limiting to {self.limit} images.' + self.bc.ENDC)
            images_list = set(itertools.islice(images_list, self.limit))
        self.download_image(folder, images_list)
        self.get_label(folder, class_name, class_code, df_val_images)
    
    def download_image(self, folder, images_list):
        '''
        Download the images.
        :param folder: train, val or test
        :param images_list: list of the images to download
        :return: None
        '''
        image_dir = folder if folder != 'val' else 'validation'
        download_dir = os.path.join(self.dataset_dir, folder, 'images')
        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir)]
        images_list = list(set(images_list) - set(downloaded_images_list))
        pool = ThreadPool(20)
        if len(images_list) > 0:
            if self.do_print:
                print(self.bc.INFO + f'Download of {len(images_list)} images in {folder}.' + self.bc.ENDC)
            commands = []
            for image in images_list:
                path = image_dir + '/' + str(image) + '.jpg' + ' ' + download_dir + '/' + str(image) + '.jpg'
                command = 'aws s3 --no-sign-request --quiet --only-show-errors cp s3://open-images-dataset/' + path
                commands.append(command)
            list(tqdm(pool.imap(os.system, commands), total=len(commands)))
            if self.do_print:
                print(self.bc.INFO + 'Done!' + self.bc.ENDC)
            pool.close()
            pool.join()
        else:
            if self.do_print:
                print(self.bc.INFO + 'All images already downloaded.' + self.bc.ENDC)
    
    def get_label(self, folder, class_name, class_code, df_val):
        '''
        Make the label.txt files
        :param folder: trai, val or test
        :param class_name: self explanatory
        :param class_code: self explanatory
        :param df_val: DataFrame values
        :return: None
        '''
        if self.do_print:
            print(self.bc.INFO + f'Creating labels for {class_name} of {folder}.' + self.bc.ENDC)
        image_dir = folder
        download_dir = os.path.join(self.dataset_dir, image_dir, 'images')
        label_dir = os.path.join(self.dataset_dir, image_dir, 'labels')

        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
        previously_downloaded_images_list = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
        images_label_list = list(set(downloaded_images_list) - set(previously_downloaded_images_list))
        groups = df_val[(df_val.LabelName == class_code)].groupby(df_val.ImageID)
        for image in images_label_list:
            try:
                current_image_path = os.path.join(download_dir, image + '.jpg')
                dataset_image = cv2.imread(current_image_path)
                boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                file_name = str(image.split('.')[0]) + '.txt'
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'w') as f:
                    for box in boxes:
                        box[0] *= int(dataset_image.shape[1])
                        box[1] *= int(dataset_image.shape[1])
                        box[2] *= int(dataset_image.shape[0])
                        box[3] *= int(dataset_image.shape[0])
                        # each row in file is name of the class_name, XMin, YMin, XMax, YMax
                        # (left top right bottom)
                        f.write(str(class_name) + ' ' + str(box[0]) + ' ' + str(box[2]) + ' ' + str(box[1]) + ' ' + str(box[3]) + '\n')
                    f.close()
            except Exception as e:
                pass
        if self.do_print:
            print(self.bc.INFO + 'Lables creation completed.' + self.bc.ENDC)
    
    def convert_annotations(self):
        classes = dict()
        yaml_path = os.path.join(ROOT_DIR, cfg.DATASET)
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.load(f.read(), Loader=yaml.Loader)
                previous_classes = data.get('names')
                for i, class_name in enumerate(previous_classes):
                    if class_name in self.classes:
                        self.classes.remove(class_name)
                    self.classes.insert(i, class_name)
        for num, class_name in enumerate(self.classes):
            classes[class_name] = num
        dataset_path = os.path.join(ROOT_DIR, cfg.DATASET_DIR)
        train_path = os.path.join(dataset_path, 'train', 'labels')
        val_path = os.path.join(dataset_path, 'val', 'labels')
        test_path = os.path.join(dataset_path, 'test', 'labels')
        train_images_path = os.path.join(dataset_path, 'train', 'images')
        val_images_path = os.path.join(dataset_path, 'val', 'images')
        test_images_path = os.path.join(dataset_path, 'test', 'images')
        labels_paths = [train_path, val_path, test_path]
        images_paths = [train_images_path, val_images_path, test_images_path]
        for label_path, image_path in zip(labels_paths, images_paths):
            try:
                for filename in os.listdir(label_path):
                    file_path = os.path.join(label_path, filename)
                    if file_path.endswith('.txt'):
                        is_already_converted = False
                        filename_str = os.path.join(image_path, str.split(str.split(file_path, '.')[-0], '/')[-1])
                        annotations = []
                        with open(file_path) as f:
                            for line in f:
                                new_line = line
                                for class_type, class_num in zip(classes.keys(), classes.values()):
                                    if class_type in line:
                                        new_line = line.replace(class_type, str(class_num))
                                        break
                                if new_line == line:
                                    is_already_converted = True
                                    break
                                labels = new_line.split()
                                coords = np.asarray([float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])])
                                coords = self.convert(filename_str, coords)
                                labels[1], labels[2], labels[3], labels[4] = coords[0], coords[1], coords[2], coords[3]
                                new_line = str(labels[0]) + " " + str(labels[1]) + " " + str(labels[2]) + " " + str(labels[3]) + " " + str(labels[4])
                                annotations.append(new_line)
                            f.close()
                        if not is_already_converted:
                            with open(file_path, 'w') as f:
                                for annotation in annotations:
                                    f.write(annotation+'\n')
                                f.close()
            except Exception as e:
                pass
    
    def convert(self, filename_str, coords):
        image = cv2.imread(filename_str + ".jpg")
        coords[2] -= coords[0]
        coords[3] -= coords[1]
        x_diff = int(coords[2]/2)
        y_diff = int(coords[3]/2)
        coords[0] = coords[0]+x_diff
        coords[1] = coords[1]+y_diff
        coords[0] /= int(image.shape[1])
        coords[1] /= int(image.shape[0])
        coords[2] /= int(image.shape[1])
        coords[3] /= int(image.shape[0])
        return coords
    
    def save_yaml(self):
        list_dir = os.listdir(self.dataset_dir)
        paths = {}
        paths['names'] = self.classes
        paths['nc'] = len(self.classes)
        for directory in list_dir:
            if directory in self.file_list_csv.keys() and directory != 'test':
                paths[directory] = os.path.join(self.dataset_dir, directory, 'images')
        with open(os.path.join(ROOT_DIR, cfg.DATASET), 'w') as f:
            yaml.dump(paths, f, default_flow_style=False)
            f.close()
        paths.update({'val': os.path.join(self.dataset_dir, 'test', 'images')})
        paths.pop('train')
        with open(os.path.join(ROOT_DIR, cfg.TEST), 'w') as f:
            yaml.dump(paths, f, default_flow_style=False)
            f.close()


if __name__ == '__main__':
    args = parser_arguments()
    downloader = Downloader(args)
    downloader.main()
    del downloader
