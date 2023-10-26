from __future__ import print_function
import os, sys
import numpy as np
from PIL import Image
import torchvision
from datasets.data_transform import imagenet_transform
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

    return images


class TinyImageNet(data.Dataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
       Args:
           root (string): Root directory of dataset where directory
               ``tiny-imagenet-200`` exists.
           train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
           transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
           target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
           download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
       """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'
    download_fname = "tiny-imagenet-200.zip"
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train, download=False, data_augment=False,
                 target_transform=None, forget_type=None, forget_num=None,
                 only_forget=False, random_exper=False):
        transform = imagenet_transform(data_augment)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.fpath = os.path.join(root, self.download_fname)
        self.target_transform = target_transform

        if download:
            self.download()

        if not check_integrity(self.fpath, self.md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        # self.classes = classes

        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root +
                                "\n Supported image extensions are: "
                                + ",".join(IMG_EXTENSIONS)))

        self.data, self.targets = [], []
        for i in range(len(data_info)):
            img_path, target = data_info[i]
            self.data.append(Image.open(img_path).convert('RGB'))
            self.targets.append(target)
        self.data = np.array(self.data)

        if random_exper:
            random_id = np.random.choice(list(range(len(self.targets))), forget_num, replace=False)
            num_class = self.num_class()
            random_target = np.random.randint(num_class, size=len(random_id))
            for i in range(forget_num):
                self.targets[random_id[i]] = random_target[i]
            self.random_id = random_id
            print("random index is {}, random target is {}".format(random_id[:5], random_target[:5]))
        if forget_type is not None:
            remove_id = []
            if forget_type == 'class':
                for i in range(len(self.targets)):
                    if self.targets[i] == forget_num:
                        remove_id.append(i)
            elif forget_type == 'random':
                remove_id = random_id
            else:
                print("error, unknown forget type!")
            if not only_forget:
                print("forget {} {} is prepared , remain data {}/{}".
                      format(forget_type, forget_num, len(self.targets) - len(remove_id), len(self.targets)))
                self.data = np.delete(self.data, remove_id, axis=0)
                self.targets = np.delete(np.array(self.targets), remove_id).tolist()
            else:
                print("only forget {} {} is prepared , remain data {}/{}".
                      format(forget_type, forget_num, len(remove_id), len(self.targets)))
                self.data = self.data[remove_id]
                self.targets = np.array(self.targets)[remove_id].tolist()
        else:
            print("all data {} is prepared !".format(len(self.data)))

    def num_class(self):
        return len(np.unique(np.array(self.targets)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
           img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

    def download(self):
        import zipfile

        if check_integrity(self.fpath, self.md5):
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.base_folder, self.md5)

        # extract file
        dataset_zip = zipfile.ZipFile(self.fpath)
        dataset_zip.extractall()
        dataset_zip.close
