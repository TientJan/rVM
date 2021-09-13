import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import scipy.ndimage
import itertools
import re

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''

def resize(img, shape):
    factors = (
        shape[0] / img.shape[0],
        shape[1] / img.shape[1],
        shape[2] / img.shape[2],
    )
    return scipy.ndimage.zoom(img, factors, mode = "constant")

def crop(img):
	x, y, _ = img.shape
	img = img[48:x-16,8:y-8, :]
	return img

def read_img(img):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img))
    img = crop(img)
    return resize(img, (160, 192,160))[np.newaxis, ...]

class Dataset(Data.Dataset):
    def __init__(self, pathnames, name_regrex = "(?P<name>Case[0-9]+)-(?P<type>[A-Z1-2_+]+).nii.gz"):
        ## 读入数据
        # pathnames:["...*t1_flair", "...*t2_flair"]
        if not isinstance(pathnames, list):
            pathnames = [pathnames]

        filelists = [glob.glob(path) for path in pathnames]  # filelists:[[...t1.nii, ...t1.nii, ...], [...t2.nii, ...t2.nii, ...]]
        self.filelists = filelists
        self.filenames = list(itertools.chain(*self.filelists))
        self.file_id = []
        self.file_type = []

        dataset = {}

        for path in self.filenames:
            # Read img & Resize
            img = read_img(path)

            # Record sub_id
            # file_id = os.path.split(os.path.split(os.path.split(path)[0])[0])[1]
            # self.file_id.append(file_id)

            # Record type
            filename = os.path.basename(path)
            # print(filename)
            m = re.search(name_regrex, filename, flags = re.IGNORECASE)
            file_type = m.group("type")
            file_id = m.group("name")
            self.file_type.append(file_type)

            ## Combine data
            if file_id not in dataset.keys():
                dataset[file_id] = {} # dataset:{'sub1':{} }
            dataset[file_id][file_type] = img # dataset:{'sub1':{"t1flair": np.array(...), "t1flair" }

        ## Conbine 2 Modals
        self.images = []
        for imageset in dataset:
            self.images.append(
                np.block([
                    dataset[imageset]["T1"],
                    dataset[imageset]["FLAIR"]
                ])
            )
            self.file_id.append(imageset)

    def __len__(self):
        # 返回数据集的大小
        return len(self.images)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        # 返回值自动转换为torch的tensor类型
        return self.images[index]