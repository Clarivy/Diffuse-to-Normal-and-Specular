import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_face_color
from PIL import Image
import numpy as np
from os.path import exists
import os
import cv2
import torch

from utils import readImage

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, mode = "input"))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            self.dir_B = opt.dataroot
            self.B_paths = sorted(make_dataset(self.dir_B, mode = opt.train_mode))
            if len(self.B_paths) != len(self.A_paths):
                raise Exception("Number of input and label not match!")
        
        if opt.isTrain and opt.face_color_transfer:
            self.opt.face_color = make_face_color(opt.face_color_path, opt)

        ### instance maps
        if not opt.no_instance:
            raise Exception("Not support instance load yet")
            exit(1)
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            raise Exception("Not support load features load yet")
            exit(1)
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = readImage(A_path, dtype=np.uint8)[:,:,:3]
        A = cv2.resize(A, (self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_CUBIC)
        params = get_params(self.opt, A)
        A = (A / 255).astype(np.float32)
        transform_A = get_transform(self.opt, params, mode = "input")
        A_tensor = transform_A(A)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            if self.opt.train_mode == "specular":
                B = readImage(B_path, dtype=np.float32)
                if len(B.shape) == 3:
                    B = B[:,:,0]
            elif self.opt.train_mode == "normal":
                B = readImage(B_path, dtype=np.float32)[:,:,:3]
            else:
                raise Exception("Error: unknown train mode")
            B = cv2.resize(B, (self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_CUBIC)
            transform_B = get_transform(self.opt, params, mode = self.opt.train_mode)
            B_tensor = transform_B(B)
            # transform_B = get_transform(self.opt, params)      

        ### if using instance maps        
        if not self.opt.no_instance:
            print("Error: not support instance yet")
            exit(1)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path, 'mask': torch.tensor(params['mask'])}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'