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
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            exit(1)
            transform_A = get_transform(self.opt, params, normalize=False) # Attention: NEARST
            A_tensor = transform_A(A)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            path1 = os.path.join(B_path, "UV_specular_merged.png")
            B1 = readImage(path1, dtype=np.uint8)

            path2 = os.path.join(B_path, "tangent.png")
            if exists(path2):
                B2 = readImage(path2, dtype=np.uint8)
            else:
                path2 = os.path.join(B_path, "total_matrix_tangent.png")
                if exists(path2):
                    B2 = readImage(path2, dtype=np.uint8)
                else:
                    print("No tangent image found!")
                    exit(1)
                
            B = cv2.merge([B1, B2[:,:,1:3]])
            B = Image.fromarray(B)
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            print("Error: not support instance yet")
            exit(1)
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'