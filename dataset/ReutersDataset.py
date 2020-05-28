"""

Contributed by Wenbin Li & Jinglin Xu

"""

import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
import scipy.io as scio
from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')

class ReutersAttrData(object):
    """
        Dataloader for Reuters datasets
    """

    def __init__(self, data_dir='/home/xujinglin/Downloads/mvdata-master/datasets/Reuters.mat',
                mode='train'):

        super(ReutersAttrData, self).__init__()

        X_list = scio.loadmat(data_dir)['X'][0].tolist()                         
        Y_list = scio.loadmat(data_dir)['Y'].tolist()             
        class_num_list = scio.loadmat(data_dir)['lenSmp'][0].tolist()     
        view_list = scio.loadmat(data_dir)['feanames'][0].tolist() 

        # Store the index of samples into the data_list
        data_list = []
        class_index_start = 0
        class_index_end = 0
        for iter, class_num in enumerate(class_num_list):
            print('The %d-th class: %d' % (iter, class_num))

            class_index_end += class_num
            sample_index = range(class_index_start, class_index_end)
            target = Y_list[class_index_start][0] 

            class_samples = []
            for i in range(len(sample_index)):
                sample_view_all = np.tile(sample_index[i], len(view_list))
                class_samples.append((sample_view_all, target))

            # divide the data into train, val and test
            random.seed(int(600)) 
            train_index = random.sample(range(0, class_num), int(0.7*class_num))
            rem_index = [rem for rem in range(0, class_num) if rem not in train_index]
            val_index = random.sample(rem_index, int(2/3.0*len(rem_index)))
            test_index = [rem for rem in rem_index if rem not in val_index]

            train_part = [class_samples[i] for i in train_index] 
            val_part = [class_samples[i] for i in val_index]
            test_part = [class_samples[i] for i in test_index]  
            
            class_index_start = class_index_end
            
            if mode == 'train':
                data_list.extend(train_part)
            elif mode == 'val':
                data_list.extend(val_part)
            else:
                data_list.extend(test_part)

        self.data_list = data_list   
        self.X_list = X_list  

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
            Load an episode each time
        """
        
        X_list = self.X_list
        (sample_view_all, target) = self.data_list[index]    
        Sample_Fea_Allviews = []

        for i in range(len(sample_view_all)):      

            sample_temp = X_list[i][sample_view_all[i], :].toarray()
            sample_temp = sample_temp[0]
            sample_temp = sample_temp.astype(float) 
            sample_temp = torch.from_numpy(sample_temp)   
            Sample_Fea_Allviews.append(sample_temp.type(torch.FloatTensor))

        return (Sample_Fea_Allviews, target)



