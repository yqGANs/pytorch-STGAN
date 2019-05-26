import numpy as np
import os
import random
import os
import sys
import json
import glob
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class Align_Celeba(data.Dataset):
    def __init__(self, config, part='train'):
        super(Align_Celeba, self).__init__()
        self.config = config
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

        img_dir = os.path.join(config.DATA_PATH)
        names = np.loadtxt(config.LABEL_PATH, skiprows=2, usecols=[0], dtype=np.str)
        self.img_paths = [os.path.join(img_dir, name) for name in names]
        att_id = [self.att_dict[att] + 1 for att in self.att_dict]
        self.labels = np.loadtxt(config.LABEL_PATH, skiprows=2, usecols=att_id, dtype=np.int64)
        if part == 'test':
            self.img_paths = self.img_paths[182637:]
            self.labels = self.labels[182637:]
        elif part == "val":
            self.img_paths = self.img_paths[182000:182637]
            self.labels = self.labels[182000:182637]
        else:
            self.img_paths = self.img_paths[:182000]
            self.labels = self.labels[:182000]

        assert len(self.img_paths) == len(self.labels)
        self.num_samples = len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.config.INPUT_SIZE[0] == 64:
            offset_h = 40
            offset_w = 15
            img_size = 128
        else:
            offset_h = 26
            offset_w = 3
            img_size = 170
        img = F.crop(img, offset_h, offset_w, img_size, img_size)
        size = (self.config.INPUT_SIZE[0], self.config.INPUT_SIZE[1])
        img = F.resize(img, size)
        img = self.img_transform(img)
        label = self.labels[index]
        label = (label + 1) // 2
        return img, label

    def __len__(self):
        return self.num_samples

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True
            )

            for item in sample_loader:
                yield item

    @staticmethod
    def generate_test_label(label, att_names, att_dicts):
        gen_label = label
        for i in range(gen_label.size(0)):
            invalid = True
            while(invalid):
                tmp_label = gen_label[i, :]
                idx = torch.randint(gen_label.size(1), (1,))
                tmp_label[idx] = 1 - tmp_label[idx]
                att_name = att_names[idx]
                if att_name in ['Bald', 'Receding_Hairline'] and tmp_label[idx] == 1:
                    tmp_label[att_dicts['Bangs']] = 0
                elif att_name == 'Bangs' and tmp_label[idx] == 1:
                    tmp_label[att_dicts['Bald']] = 0
                    tmp_label[att_dicts['Receding_Hairline']] = 0
                elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and tmp_label[idx] == 1:
                    for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair',  'Gray_Hair']:
                        if n != att_name:
                            tmp_label[att_dicts[n]] = 0
                    tmp_label[att_dicts['Bald']] = 0
                elif att_name in ['Straight_Hair', 'Wavy_Hair'] and tmp_label[idx] == 1:
                    for n in ['Straight_Hair', 'Wavy_Hair']:
                        if n != att_name:
                            tmp_label[att_dicts[n]] = 0
                elif att_name in ['Mustache', 'No_Beard'] and tmp_label[idx] == 1:
                    for n in ['Mustache', 'No_Beard']:
                        if n != att_name:
                            tmp_label[att_dicts[n]] = 0
                if not torch.equal(gen_label[i, :], tmp_label):
                    invalid = False
                    gen_label[i, :] = tmp_label
        return gen_label



if __name__ == "__main__":
    import yaml
    import argparse
    class Config(dict):
        def __init__(self, config_path):
            super(Config, self).__init__()
            with open(config_path, 'r') as f:
                self._yaml = f.read()
                self._dict = yaml.load(self._yaml)
                self._dict['PATH'] = os.path.dirname(config_path)

        def __getattr__(self, name):
            if self._dict.get(name) is not None:
                return self._dict[name]

            return None

        def print(self):
            print('Model configurations:')
            print('---------------------------------')
            print(self._yaml)
            print('')
            print('---------------------------------')
            print('')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/STGAN.yml',
                        help='training configure path (default: ./configs/STGAN.yml)')
    args = parser.parse_args()
    config = Config(args.config_path)
    test_dataset = Align_Celeba(config, 'test')
    print(len(test_dataset))
    print(test_dataset.labels.shape)
    print(test_dataset.labels[5])
