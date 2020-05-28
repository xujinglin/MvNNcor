"""

Contributed by Wenbin Li & Jinglin Xu

"""

from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from torch import autograd
from PIL import ImageFile

from dataset.AWADataset import animalAttrData
import models.network_MvNNcor as MultiviewNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Load the pre-trained model
data_name = 'AWA'
data_dir = './mvdata/Animals_with_Attributes/Features'
model_trained = './results/MvNNcor_AWA_multiviewNet_Epochs_50_fea_200_300_6.0_64/model_best.pth.tar'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=data_dir, help='the path of data')
parser.add_argument('--data_name', default=data_name, help='The name of the data')
parser.add_argument('--mode', default='test', help='train|val|test')
parser.add_argument('--outf', default='./results/MvNNcor')
parser.add_argument('--resume', default=model_trained, help='use the saved model')
parser.add_argument('--basemodel', default='multiviewNet', help='multiviewNet')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=64, help='the mini-batch size of training')
parser.add_argument('--testSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--num_classes', type=int, default=50, help='the number of classes')
parser.add_argument('--num_view', type=int, default=6, help='the number of views')
parser.add_argument('--fea_out', type=int, default=200, help='the dimension of the first linear layer')
parser.add_argument('--fea_com', type=int, default=300, help='the dimension of the combination layer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gamma', type=float, default=6.0, help='the power of the weight for each view')

opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# save the opt and results to txt file
opt.outf = opt.outf+'_'+opt.data_name+'_Epochs_'+str(opt.epochs)+'_'+str(opt.gamma)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

txt_save_path = os.path.join(opt.outf, 'test_results.txt')
F_txt = open(txt_save_path, 'a+')

# ======================================== Folder of Datasets ==========================================
if opt.data_name == 'AWA':
    testset = animalAttrData(data_dir=opt.dataset_dir, mode=opt.mode)

elif opt.data_name == 'Hand':
    testset = HandwrittenAttrData(data_dir=opt.dataset_dir, mode=opt.mode)

elif opt.data_name == 'Caltech101-all':
    testset = Caltech101AttrData(data_dir=opt.dataset_dir, mode=opt.mode)

elif opt.data_name == 'Mnist':
    testset = MNISTAttrData(data_dir=opt.dataset_dir, mode=opt.mode)

elif opt.data_name == 'NUS':
    testset = NUSWIDEAttrData(data_dir=opt.dataset_dir, mode=opt.mode)

elif opt.data_name == 'Reuter':
    testset = ReutersAttrData(data_dir=opt.dataset_dir, mode=opt.mode)
else:
    print('There is something wrong!')

print('Testset: %d' %len(testset))
print('Testset: %d' %len(testset), file=F_txt)

# ========================================== Load Datasets ==============================================
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=opt.testSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    ) 
print(opt)
print(opt, file=F_txt)

# ========================================== Model config ===============================================
global best_prec1, epoch, weight_var
best_prec1 = 0
epoch = 0
weight_var = torch.zeros(opt.num_view)   
weight_var = weight_var.to("cuda")

test_iter = iter(test_loader)
testdata, target = test_iter.next()
view_list = []
for v in range(len(testdata)):
    temp_size = testdata[v].size()
    view_list.append(temp_size[1])

ngpu = int(opt.ngpu)
model = MultiviewNet.define_MultiViewNet(which_model=opt.basemodel, norm='batch', init_type='normal', 
    use_gpu=opt.cuda, num_classes=opt.num_classes, num_view=opt.num_view, view_list=view_list,
    fea_out=opt.fea_out, fea_com=opt.fea_com)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        weight_var = checkpoint['weight_var']
        model.load_state_dict(checkpoint['state_dict'])
        # print(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

if opt.ngpu > 1:
    model = nn.DataParallel(model, range(opt.ngpu))

print(model) 
print(model, file=F_txt)

# ======================================= Define functions =============================================
def validate(val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        end = time.time()

        for index, (sample_set, sample_targets) in enumerate(val_loader):

            input_var = [sample_set[i].cuda() for i in range(len(sample_set))]

            # deal with the target
            target_var = sample_targets.cuda()

            Output_list = model(input_var)
            loss = torch.zeros(1).to("cuda")
            
            # pdb.set_trace()
            for v in range(len(Output_list)):
                loss_temp = criterion(Output_list[v], target_var)
                loss += (weight_var[v] ** gamma) * loss_temp

            output_var = torch.stack(Output_list)
            weight_var = weight_var.unsqueeze(1)
            weight_var = weight_var.unsqueeze(2)
            weight_var = weight_var.expand(weight_var.size(0), opt.batchSize, opt.num_classes)
            output_weighted = weight_var * output_var
            output_weighted = torch.sum(output_weighted, 0)

            weight_var = weight_var[:,:,1]
            weight_var = weight_var[:,1]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output_weighted, target_var, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            top5.update(prec5[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % opt.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, index, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5))

                print('Test: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, index, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1, top5=top5), file=F_txt)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top5=top5, best=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1, top5=top5, best=best_prec1), file=F_txt)
        print(weight_var)
        print(weight_var, file=F_txt)
    
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ============================================ Testing phase ========================================
print('start testing.........')
start_time = time.time()  
gamma = torch.tensor(opt.gamma).to("cuda")
prec2 = validate(test_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)
F_txt.close()

# ============================================ Testing End ========================================
