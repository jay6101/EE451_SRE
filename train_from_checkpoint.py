import os
import shutil
import time
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import data
import losses
import ramps
import label_unlabel_indexes

warnings.filterwarnings('ignore')

print("importing done")

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()

#if not use_gpu:
#    raise ValueError("Error, requires GPU")

print("Available GPU count:" + str(gpu_count)) 

def checkpoint(model, ema_model, best_loss, epoch, optimizer):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """
    #print('saving')
    for param_group in optimizer.param_groups:
        LR = param_group['lr']

    state = {
        'model': model,
        'ema_model': ema_model,
        'best_loss': best_loss,
        'epoch': epoch,
        'LR': LR
    }

    torch.save(state, '/home/jay/sre/results/checkpoint_{}'.format(epoch))

def create_model(ema=False):

    num_classes = 14
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs,num_classes), nn.Sigmoid())
    #model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def adjust_learning_rate(optimizer):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/1.5

    return optimizer

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100 * ramps.sigmoid_rampup(epoch, 5)

cudnn.benchmark = True

checkpoint_ = torch.load('/home/jay/sre/results/checkpoint_43')

NUM_EPOCHS = 100
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
N_LABELS = 14  # we are predicting 14 labels
LR = checkpoint_['LR']

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_text_file = "/home/jay/sre/train_list.txt"
labeled_train_text_file = "/home/jay/sre/labeled_train_list.txt"
val_text_file = "/home/jay/sre/val_list.txt"


labeled_idxs, unlabeled_idxs = label_unlabel_indexes.label_unlabeled_idxs(train_text_file,labeled_train_text_file)

batch_size = 16
labeled_batch_size = 4
batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, batch_size , labeled_batch_size)

train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0, hue=0),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    	])

val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
train_dataset = data.ChestXrayDataSet(data_dir='/home/jay/sre/images',
                            image_list_file=train_text_file,
                            labelled_list_file = labeled_train_text_file,
                            transform=train_transform,
                            train=True
                            )

val_dataset = data.ChestXrayDataSet(data_dir='/home/jay/sre/images',
                            image_list_file=val_text_file,
                            labelled_list_file = labeled_train_text_file,
                            transform=val_transform,
                            train=False
                            )
#end = time.time()
#print(end-start)
if use_gpu:
	train_loader = DataLoader(dataset=train_dataset, batch_sampler = batch_sampler,
                         num_workers=4, pin_memory=True)

	val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)
else:
	train_loader = DataLoader(dataset=train_dataset, batch_sampler = batch_sampler,
                         num_workers=0, pin_memory=True)

	val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                         num_workers=0, pin_memory=True, drop_last=False)

#model = create_model()
#ema_model = create_model(ema=True)

model = checkpoint_['model']
ema_model = checkpoint_['ema_model']

if use_gpu:
	model = model.cuda()
	ema_model = ema_model.cuda()



optimizer = torch.optim.SGD(
	filter(lambda p: p.requires_grad,model.parameters()),
		lr=LR,
		momentum=0.9,
		weight_decay=WEIGHT_DECAY)

class_criterion = nn.BCELoss()
consistency_criterion = losses.softmax_mse_loss

start_epoch = checkpoint_['epoch']+1
val_prev_loss = checkpoint_['best_loss']
global_step = int(len(train_dataset)/BATCH_SIZE)*(start_epoch) 
alpha = 0.99

print("Training started....")

for epoch in range(start_epoch, NUM_EPOCHS):

    model.train()
    ema_model.train()

    running_loss = 0.0
    running_class_loss = 0.0
    running_ema_class_loss  = 0.0
    running_consistency_loss = 0.0

    for i,(inputs,ema_inputs,labels) in enumerate(tqdm(train_loader)):

        for x in range(len(inputs)):
            if labels[x][0]!= -1:
                break

        if use_gpu:
        	input_var = torch.autograd.Variable(inputs.cuda())
        	ema_input_var = torch.autograd.Variable(ema_inputs.cuda())
        	labels = torch.autograd.Variable(labels.cuda())
        else:
        	input_var = torch.autograd.Variable(inputs)
        	ema_input_var = torch.autograd.Variable(ema_inputs)
        	labels = torch.autograd.Variable(labels)

        batch_size = len(labels)

        ema_model_out = ema_model(ema_input_var)
        model_logit = model(input_var)

        ema_logit = Variable(ema_model_out.detach().data, requires_grad=False)

        consistency_weight = get_current_consistency_weight(epoch)
        consistency_loss = consistency_weight * consistency_criterion(model_logit, ema_logit) / batch_size

        class_loss = class_criterion(model_logit[x:], labels[x:]) / batch_size
        ema_class_loss = class_criterion(ema_logit[x:], labels[x:]) / batch_size

        loss = class_loss + consistency_loss

        #prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        #ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step +=1

        running_loss += loss.item() * batch_size
        running_class_loss += class_loss.item() * batch_size
        running_ema_class_loss += ema_class_loss.item()*batch_size
        running_consistency_loss += consistency_loss.item() * batch_size

        #print("running loss: {}".format(running_loss))
    
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)

    epoch_loss = running_loss #/ len(train_dataset)
    epoch_class_loss = running_class_loss #/ len(train_dataset)
    epoch_ema_class_loss = running_ema_class_loss #/ len(train_dataset)
    epoch_cons_loss = running_consistency_loss #/ len(train_dataset) 

    print("EPOCH: {}".format(epoch))
    print('Total loss {}'.format( epoch_loss))
    print("class loss {}".format( epoch_class_loss))
    print("ema class loss {}".format( epoch_ema_class_loss))
    print('consistency loss {}'.format(epoch_cons_loss))

    ################# Both Model's evaluation####################################
    model.eval()
    ema_model.eval()

    running_eval_loss = 0.0
    running_eval_class_loss = 0.0
    running_eval_ema_class_loss = 0.0
    running_eval_consistency_loss = 0.0
    ground_truths = np.array([1]*N_LABELS).reshape(1,N_LABELS)
    predictions = np.array([1]*N_LABELS).reshape(1,N_LABELS)

    for i,(inputs,ema_inputs,labels) in enumerate(val_loader):

    	if use_gpu:
    		input_var = torch.autograd.Variable(inputs.cuda())
    		ema_input_var = torch.autograd.Variable(inputs.cuda())
    		labels = torch.autograd.Variable(labels.cuda())
    	else:
    		input_var = torch.autograd.Variable(inputs)
    		ema_input_var = torch.autograd.Variable(inputs)
    		labels = torch.autograd.Variable(labels)

    	batch_size = len(labels)

    	ema_model_logit = ema_model(ema_input_var)
    	model_logit = model(input_var)

    	class_eval_loss = class_criterion(model_logit, labels) / batch_size
    	ema_class_eval_loss = class_criterion(ema_model_logit, labels) / batch_size
    	consistency_eval_loss = get_current_consistency_weight(epoch)*consistency_criterion(model_logit, ema_model_logit) / batch_size

    	running_eval_loss += (class_eval_loss+consistency_eval_loss).item() * batch_size
    	running_eval_class_loss += class_eval_loss.item() * batch_size
    	running_eval_ema_class_loss += ema_class_eval_loss.item() * batch_size
    	running_eval_consistency_loss += consistency_eval_loss.item() * batch_size

    	gt = labels.cpu().data.numpy()
    	pred = ema_model_logit.cpu().data.numpy()

    	ground_truths = np.concatenate((ground_truths,gt),axis=0)
    	predictions = np.concatenate((predictions,pred),axis=0)

    auc_score = roc_auc_score(ground_truths[1:,:],predictions[1:,:])

    vepoch_loss = running_eval_loss #/ len(val_dataset)
    vepoch_class_loss = running_eval_class_loss #/ len(val_dataset)
    vepoch_ema_class_loss = running_eval_ema_class_loss# / len(val_dataset)
    vepoch_cons_loss = running_eval_consistency_loss# / len(val_dataset)

    print('Total val loss {}'.format(vepoch_loss))
    print("class val loss {}".format(vepoch_class_loss))
    print("ema class val loss {}".format(vepoch_ema_class_loss))
    print('consistency val loss {}'.format(vepoch_cons_loss))
    print('AUC score: {}'.format(auc_score))

    checkpoint(model,ema_model,vepoch_loss,epoch,optimizer)

    if val_prev_loss<=vepoch_loss:
    	optimizer = adjust_learning_rate(optimizer)
    val_prev_loss = vepoch_loss

    with open(r'sre/train_log.txt', 'a+') as f:
    	f.write("EPOCH: {} \n".format(epoch))
    	f.write('Total loss {} \n'.format(epoch_loss))
    	f.write("class loss {} \n".format(epoch_class_loss))
    	f.write("ema class loss {} \n".format(epoch_class_loss))
    	f.write('consistency loss {} \n'.format(epoch_cons_loss))
    	f.write('Total val loss {} \n'.format(vepoch_loss))
    	f.write("class val loss {} \n".format(vepoch_class_loss))
    	f.write("ema class val loss {} \n".format(vepoch_ema_class_loss))
    	f.write('consistency val loss {} \n'.format(vepoch_cons_loss))
    	f.write('AUC score: {} \n'.format(auc_score))
    	f.write("\n")
    	f.write("\n")





     





