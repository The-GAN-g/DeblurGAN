import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import matplotlib.pyplot as plt
# import time
# from torch.autograd import Variable

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    Get paths of the sharp/blurred image to
    separate them
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_transform():
    """
    Apply standard transformation on all
    the images
    """
    transform_list = []
    transform_list += [transforms.Resize([256, 256]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class CreateDataset(data.Dataset):
    """
    Creates train and test data-set
    """
    def __init__(self):
        super(CreateDataset, self).__init__()

        dataroot = 'images/'
        phase = 'train/'  # train for training and test for testing
        self.dir_A = os.path.join(dataroot, phase, 'A')  # directory for blur images
        self.dir_B = os.path.join(dataroot, phase, 'B')  # directory for sharp images
        self.A_paths = make_dataset(self.dir_A)  # get paths of all blurred images
        self.B_paths = make_dataset(self.dir_B)  # get paths of all sharp images
        self.A_paths = sorted(self.A_paths)  # sort the images because all the related images are together
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)  # 2103 images in A and B each
        self.B_size = len(self.B_paths)
        self.transform = get_transform()  # apply transforms


    def name(self):
        return 'BaseDataset'


    def __getitem__(self, index):  # not used
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        B_path = self.B_paths[index % self.B_size]
        index_B = index % self.B_size
        #        print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        return max(self.A_size, self.B_size)


class CreateDataLoader():
    """
    Create a data loader to fetch images
    """
    def name(self):
        return 'CreateDataLoader'


    def __init__(self, batchSize):
        super(CreateDataLoader, self).__init__()
        #         batchSize = 1
        self.dataset = CreateDataset()  # Call to create dataset class
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=False
        )


    def load_data(self):
        return self.dataloader


    def __len__(self):
        return len(self.dataset)