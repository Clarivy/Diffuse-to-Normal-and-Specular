import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from os.path import exists
import os
import cv2

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
            self.B_paths = sorted(make_dataset(self.dir_B, mode = "label"))

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
        A = readImage(A_path, dtype=np.float32)
        params = get_params(self.opt)
        transform_A = get_transform(self.opt, params, mode = "input")
        A_tensor = transform_A(A)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            path1 = os.path.join(B_path, "UV_specular_merged.png")
            B1 = readImage(path1, dtype=np.float32)

            path2 = os.path.join(B_path, "tangent.png")
            if exists(path2):
                B2 = readImage(path2, dtype=np.float32)
            else:
                path2 = os.path.join(B_path, "total_matrix_tangent.png")
                if exists(path2):
                    B2 = readImage(path2, dtype=np.float32)
                else:
                    print("No tangent image found!")
                    exit(1)
                
            B = cv2.merge([B1, B2[:,:,0:2]])
            transform_B = get_transform(self.opt, params, mode = "label")
            B_tensor = transform_B(B)
            # transform_B = get_transform(self.opt, params)      

        ### if using instance maps        
        if not self.opt.no_instance:
            print("Error: not support instance yet")
            exit(1)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'