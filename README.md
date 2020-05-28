# MvNNcor
This is an implementation of Deep Embedded Complementary and Interactive Information for Multi-view Classification (MvNNcor) in Pytorch.

## Requirements

## Datasets
The model is trained on AWA/Caltech101/Reuters dataset, where each dataset are splited into three parts: 70% samples for training, two-thirds of the rest samples for validation, and one-third of that for testing. We utilize the classification accuracy to evaluate the performance of all the methods.

## Implementation
Taking AWA dataset as an example,

* Training
[python MvNNcor_Train.py --dataset_dir ./mvdata/AWA/Features --data_name AWA --num_classes 50 --num_view 6 --gamma 6.0]

* Testing 
[python MvNNcor_Test.py --dataset_dir ./mvdata/AWA/Features --data_name AWA --resume ./results/.../model_best.pth.tar --num_classes 50 --num_view 6 --gamma 6.0]

For different datasets, you need to use different parameters refering to the experimental settings of our AAAI2020 paper.

## Citation
Deep Embedded Complementary and Interactive Information for Multi-view Classification. AAAI2020
