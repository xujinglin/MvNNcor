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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/xujinglin/Documents/DataSets/Animals_with_Attributes/Features', help='the path of data')
parser.add_argument('--data_name', default='AWA', help='The name of the data')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/MvNNcor')
parser.add_argument('--net', default='', help='use the saved model')
parser.add_argument('--basemodel', default='multiviewNet', help='multiviewNet')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=64, help='the mini-batch size of training')
parser.add_argument('--testSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
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
opt.outf = opt.outf+'_'+opt.data_name+'_'+str(opt.basemodel)+'_Epochs_'+str(opt.epochs)+'_fea_'+str(opt.fea_out)+'_'+str(opt.fea_com)+'_'+str(opt.gamma)+'_'+str(opt.batchSize)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

txt_save_path = os.path.join(opt.outf, 'opt_results.txt')
F_txt = open(txt_save_path, 'a+')


# ======================================== Folder of Datasets ==========================================

if opt.data_name == 'AWA':

    trainset = animalAttrData(data_dir=opt.dataset_dir, mode=opt.mode)
    valset = animalAttrData(data_dir=opt.dataset_dir, mode='val')
    testset = animalAttrData(data_dir=opt.dataset_dir, mode='test')

print('Trainset: %d' %len(trainset))
print('Valset: %d' %len(valset))
print('Testset: %d' %len(testset))
print('Trainset: %d' %len(trainset), file=F_txt)
print('Valset: %d' %len(valset), file=F_txt)
print('Testset: %d' %len(testset), file=F_txt)


# ========================================== Load Datasets ==============================================

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=opt.batchSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    )
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=opt.testSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    ) 
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=opt.testSize, shuffle=True, 
    num_workers=int(opt.workers), drop_last=True, pin_memory=True
    ) 
print(opt)
print(opt, file=F_txt)


# ========================================== Model config ===============================================

train_iter = iter(train_loader)
traindata, target = train_iter.next()    
view_list = []
for v in range(len(traindata)):
    temp_size = traindata[v].size()
    view_list.append(temp_size[1])       

ngpu = int(opt.ngpu)
model = MultiviewNet.define_MultiViewNet(which_model=opt.basemodel, norm='batch', init_type='normal', 
    use_gpu=opt.cuda, num_classes=opt.num_classes, num_view=opt.num_view, view_list=view_list,
    fea_out=opt.fea_out, fea_com=opt.fea_com)

if opt.net != '': 
    model.load_state_dict(torch.load(opt.net))

if opt.ngpu > 1:
    model = nn.DataParallel(model, range(opt.ngpu))

print(model) 
print(model, file=F_txt)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

# ======================================= Define functions =============================================

def reset_grad():
    model.zero_grad()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.05 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch, F_txt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for index, (sample_set, sample_targets) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = [sample_set[i].cuda() for i in range(len(sample_set))]

        # deal with the target
        target_var = sample_targets.to("cuda")

        # compute output
        Output_list = model(input_var)      

        weight_up_list = []
        loss = torch.zeros(1).to("cuda")

        for v in range(len(Output_list)):
            loss_temp = criterion(Output_list[v], target_var)
            loss += (weight_var[v] ** gamma) * loss_temp

            weight_up_temp = loss_temp ** (1/(1-gamma))
            weight_up_list.append(weight_up_temp)   

        output_var = torch.stack(Output_list)

        weight_var = weight_var.unsqueeze(1)
        weight_var = weight_var.unsqueeze(2)
        weight_var = weight_var.expand(weight_var.size(0), opt.batchSize, opt.num_classes)
        output_weighted = weight_var * output_var
        output_weighted = torch.sum(output_weighted, 0)

        weight_var = weight_var[:,:,1]
        weight_var = weight_var[:,1]
        weight_up_var = torch.FloatTensor(weight_up_list).to("cuda")
        weight_down_var = torch.sum(weight_up_var)
        weight_var = torch.div(weight_up_var, weight_down_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_weighted, target_var, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))
        top5.update(prec5[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % opt.print_freq == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, index, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5), file=F_txt)

    return weight_var

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
    
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        file_model_best = os.path.join(opt.outf, 'model_best.pth.tar')
        shutil.copyfile(filename, file_model_best)

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

# ============================================ Training phase ========================================

print('start training.........')
start_time = time.time()
best_prec1 = 0
weight_var = torch.ones(opt.num_view) * (1/opt.num_view)   
weight_var = weight_var.to("cuda")
gamma = torch.tensor(opt.gamma).to("cuda")

for epoch in range(opt.epochs):
    # adjust the learning rate
    adjust_learning_rate(optimizer, epoch) 

    # train for one epoch
    weight_var = train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch, F_txt)

    # evaluate on validation/test 
    print('=============== Testing in the validation set ===============')    
    prec1 = validate(val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)  


    print('================== Testing in the test set ==================')   
    prec2 = validate(val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt)  

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    # save the checkpoint
    filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' %epoch)
  

    if is_best:
        save_checkpoint(
        {
            'epoch': epoch + 1,
            'arch': opt.basemodel,
            'state_dict': model.state_dict(),
            'weight_var': weight_var,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)


print('======== Training END ========')
F_txt.close()

# ============================================ Training End ========================================
