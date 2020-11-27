
import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import PIL
from glob import glob
from unet import unet
from utils import *
from PIL import Image

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = glob(dir+"/*.png")+glob(dir+"/*.jpg")+glob(dir+"/*.jpeg")
   
    return images

class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def img_test(self, image):
        self.imsize = int((max(image.size)/32))*32
        print(self.imsize)
        transform = transformer(True, True, True, False, self.imsize) 
        
        img = transform(image).to(self.device)
        img = img.unsqueeze(0)
        labels_predict = self.G(img)
        labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        labels_predict_color = generate_label(labels_predict, self.imsize)
        return labels_predict_color[0], labels_predict_plain[0]

    def test(self):
        transform = transformer(True, True, True, False, self.imsize) 
        test_paths = make_dataset(self.test_image_path)
        make_folder(self.test_label_path, '')
        make_folder(self.test_color_label_path, '') 
        batch_num = int(self.test_size / self.batch_size)
        imgs = []
        for j in range(len(test_paths)):
            path = test_paths[j]
            img = transform(Image.open(path))
            imgs.append(img)
            imgs = torch.stack(imgs) 
            # imgs = imgs.cuda()
            imgs = imgs.to(self.device)
            labels_predict = self.G(imgs)
            labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
            labels_predict_color = generate_label(labels_predict, self.imsize)
            cv2.imwrite(os.path.join(self.test_label_path, test_paths[j].split("/")[-1].split(".")[0] +'.png'), labels_predict_plain[0])
            save_image(labels_predict_color[0], os.path.join(self.test_color_label_path, test_paths[j].split("/")[-1].split(".")[0] +'.png'))

    def build_model(self):
        self.G = unet().to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
        if(torch.cuda.is_available()):
            self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.model_name)))
        else:
            self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.model_name), map_location=torch.device('cpu')))
        self.G.eval() 
        # print networks
        print("Model Loaded!")
