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
from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')



class animalAttrData(object):
    """
       Dataloader for animal attributes dataset.
    """

    def __init__(self, data_dir='/home/xujinglin/Documents/DataSets/Animals_with_Attributes/Features', 
                mode='train'):
        
        super(animalAttrData, self).__init__()

        data_list = []
        fea_name = os.listdir(data_dir)
        class_name = os.listdir(os.path.join(data_dir, fea_name[0]))

        count = -1   
        for class_item in class_name:

            count += 1
            class_list = []
            class_path_list = []
            for fea_item in fea_name:
                class_path_list.append(os.path.join(data_dir, fea_item, class_item))

            sample_name = os.listdir(class_path_list[0])

            # each sample have servel kinds of features
            for sample_item in sample_name:
                
                sample_fea_all = [os.path.join(class_path_list[i], sample_item) for i in range(len(class_path_list))]
                class_list.append((sample_fea_all, count))
            
            # divide the data into training set and testing set
            random.seed(int(100)) 
            train_part = random.sample(class_list, int(0.7*len(class_list)) )
            rem_part = [rem for rem in class_list if rem not in train_part]
            val_part = random.sample(rem_part, int(2/3.0*len(rem_part)))
            test_part = [te for te in rem_part if te not in val_part]


            if mode == 'train':
                data_list.extend(train_part)
            elif mode == 'val':
                data_list.extend(val_part)
            else:
                data_list.extend(test_part)

        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time
        '''
        
        (sample_fea_all, target) = self.data_list[index]
        Sample_Fea = []
        for i in range(len(sample_fea_all)):
            fea_temp = np.loadtxt(sample_fea_all[i]) 
            fea_temp = torch.from_numpy(fea_temp)
            Sample_Fea.append(fea_temp.type(torch.FloatTensor))

        return (Sample_Fea, target)
        

