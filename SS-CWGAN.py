# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:22:25 2020

@author: lbx
"""


import argparse
import os

import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.utils.data as Data
from scipy.io import loadmat
from scipy.io import savemat
import copy
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt_sne
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import plotly.express as px

seednum = 8

torch.manual_seed(seednum)
UnlabeledData_batchsize = 128

# *********************** Step 1: the dataset *********************************
# filepath = "C:/Users\HNU\Desktop\对比\KNN_Iris-Machine_Learning-master\KNN_Iris-Machine_Learning-master\ProcessData40.mat"
#
# labeled_data = loadmat(filepath)['LabeledSetNor']
# labels = loadmat(filepath)['SupLabel'].flatten()
# # x_led = labeled_data
# # y_led = labels
# x_led = []
# y_led = []
# for i in range(8):
#     x_l = labeled_data[labels == i, :][:13, :]
#     y_l = labels[labels == i][:13]
#     x_led.append(x_l)
#     y_led.append(y_l)
# x_led = torch.tensor(x_led)
# labeled_data = x_led.reshape(-1,61).float()
# labels = torch.Tensor(y_led).flatten()
# labels = torch.tensor(labels,dtype=torch.int64)
transform = transforms.Compose([transforms.ToTensor()])
filepath = "D:\semi_gan\ScienceDirect_files_29Nov2023_02-01-20.166/1-s2.0-S030626192100026X-mmc4\ProcessData120.mat"
# Load the dataset
labeled_data = loadmat(filepath)['LabeledSetNor']
labels = loadmat(filepath)['SupLabel']
labeled_data = torch.from_numpy(labeled_data).float()
labels = torch.from_numpy(labels).squeeze().long()

unlabeleddata = loadmat(filepath)['UnlabeledSetNor']
unlabeleddata = torch.from_numpy(unlabeleddata).float()

testdata = loadmat(filepath)['TestSetNor']
testlabel = loadmat(filepath)['TestLabel']
testdata = torch.from_numpy(testdata).float()
testlabel = torch.from_numpy(testlabel).squeeze().long()

# create the data loader
unlabeledset = Data.TensorDataset(unlabeleddata)
testset = Data.TensorDataset(testdata, testlabel)

UnlabeledData_trainloader = torch.utils.data.DataLoader(unlabeledset, batch_size=UnlabeledData_batchsize, shuffle=True)
def one_hot(y, size, reuse=False):
    label = []
    for i in range(size):
        a = int(y[i])
        temp = [0, 0, 0, 0, 0, 0, 0, 0]
        temp[a] = 1
        if i == 0:
            label = temp
        else:
            label.extend(temp)
    label = np.array(label).reshape(size, 8)
    return label

# ****************** Step 2: Define the network (generator) *******************
class Generator(torch.nn.Module):
    def __init__(self, n_randomcode, n_hidden1, n_hidden2, n_output):
        super(Generator, self).__init__()
        self.hidden1 = torch.nn.Linear(n_randomcode, n_hidden1)
        self.L1 = nn.LeakyReLU(0.3)
        # self.b1 = nn.BatchNorm1d(n_hidden1)
        # self.batchsize(n_hidden2)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.L2 = nn.LeakyReLU(0.3)
        # self.b2 = nn.BatchNorm1d(n_hidden2)
        self.output = torch.nn.Linear(n_hidden2, n_output)
        
    def forward(self,x,y):
        x = torch.cat((x,y),dim=1)
        x = self.hidden1(x)
        x = self.L1(x)
        # x = self.b1(x)
        x = self.hidden2(x)
        x = self.L2(x)
        # x = self.b2(x)
        x = F.tanh(self.output(x))
        return x

n_randomcode = 16

# ****************** Step 3: Define the network (discriminator) ***************
class Discriminator(torch.nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_output):
        super(Discriminator, self).__init__()
        self.hidden1 = torch.nn.Linear(n_features, n_hidden1)
        self.L3 = nn.LeakyReLU(0.3)
        # self.b3 = nn.BatchNorm1d(n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.L4 = nn.LeakyReLU(0.3)
        # self.b4 = nn.BatchNorm1d(n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        self.w = torch.nn.Linear(n_output,1)
        
    def forward(self,x):
        x = self.hidden1(x)
        x = self.L3(x)
        # x = self.b3(x)
        x = self.hidden2(x)
        x = self.L4(x)
        # x = self.b4(x)
        x = self.predict(x)
        x1 = self.w(x)
        return x, x1

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = torch.tensor((alpha * real_samples + ((1 - alpha) * fake_samples)),dtype=torch.float32).requires_grad_(True)
    _,d_interpolates = D(interpolates)
    # interpolates = torch.tensor(interpolates.view(interpolates.shape[0], -1),dtype= torch.float32).requires_grad_(True)
    # d_interpolates = F.softplus(log_sum_exp(D(interpolates)))
    # fake = log_sum_exp(D(interpolates))
    fake = torch.ones_like(d_interpolates).requires_grad_(False)
    # fake = Variable((real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ****************** Step 4: Start to train the network ***********************
netD = Discriminator(61, 32, 16, 8)
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0009, betas=(0.5,0.999))

netG = Generator(n_randomcode, 64, 128, 61)
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0009 , betas=(0.5,0.999))

EPOCH = 500
l_his = []
best_model_wts = copy.deepcopy(netD.state_dict())
best_acc = 0.0
lambda_gp = 10

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    current_loss = 0
    
    for step, unlabeled_sample in enumerate(UnlabeledData_trainloader):
        netD.train()
        optimizerD.zero_grad()
        
        # ----------------- train the discriminator ---------------------    
        # 1. on Unlabeled data
        unlabeled_sample = unlabeled_sample[0]
        output,_ = netD(unlabeled_sample)
        logsumexp_output = log_sum_exp(output)
        unlabeled_loss = 0.5*(torch.mean(F.softplus(logsumexp_output)) - torch.mean(logsumexp_output))
        
            
        # 2. on fake data (generated by generator network)
        lg = np.random.randint(0, 8, len(unlabeled_sample))
        l_g = torch.from_numpy(one_hot(lg, len(unlabeled_sample)))
        noise = torch.randn(len(unlabeled_sample), 8)
        generated = netG(noise,l_g)
        output1,_ = netD(generated.detach())
        logsumexp_fake = log_sum_exp(output1)
        fake_loss = 0.5*torch.mean(F.softplus(logsumexp_fake))
        gradient_penalty = compute_gradient_penalty(netD, unlabeled_sample.data, generated.data)
        
        
        # 3. on Labeled data            
        output,_ = netD(labeled_data)
        logsumexp_output = log_sum_exp(output)
        predicted_label = torch.gather(output, 1, labels.unsqueeze(1))
        labeled_loss = torch.mean(logsumexp_output) - torch.mean(predicted_label)
        # label_loss2 = 0.5*(torch.mean(F.softplus(logsumexp_output)) - torch.mean(logsumexp_output))
        
        
        loss =   labeled_loss + unlabeled_loss + fake_loss + lambda_gp*gradient_penalty
        loss.backward()
        optimizerD.step()
        
        # ----------------- train the generator ----------------------
        netG.train()
        optimizerG.zero_grad()
        
        output,_ = netD(generated)
        logsumexp_output = log_sum_exp(output)
        lossG = 0.5*(torch.mean(F.softplus(logsumexp_output)) - torch.mean(logsumexp_output))
        lossG.backward()
        optimizerG.step()
        
    # accuracy on the testing set
    netD.eval()
    with torch.no_grad():
        ps,_ = netD(testdata)
        ps = torch.exp(ps)
    
    ps = ps.numpy()
    pred_label = np.argmax(ps, axis=1)
    true_label = testlabel.numpy()

    accuracy_test = np.sum(pred_label == true_label)/testlabel.shape[0]
    if epoch > 10:
        l_his.append(accuracy_test)
    
    if accuracy_test > best_acc:
        best_acc = accuracy_test
        best_model_wts = copy.deepcopy(netD.state_dict())
        best_epoch_index = epoch
    
    print("Epoch {} - Testing loss: {}".format(epoch, accuracy_test))
    
plt.plot(l_his)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
print('Best test accuracy: ', best_acc*100, '% after ', best_epoch_index, ' training epochs')
plt.show()
f_p = pred_label
f_t = true_label
print("F1-Score:{:.4f}".format(f1_score(f_t,f_p,average='micro')))
tm = np.array(l_his)
tm = tm[480:]
a = np.std(tm)
print("标准差：{:.4f}", a)
print("混淆矩阵:{:.4f}",confusion_matrix(f_p, f_t)/len(f_p), '%')
def confusion_matrix(preds,labels,conf_matrix):
    for p,t in zip(preds,labels):
        conf_matrix[p,t] +=1
    return conf_matrix
conf_matrix = torch.zeros(8,8)
conf_matrix = confusion_matrix(f_p, f_t,conf_matrix)
conf_matrix.cpu()
conf_matrix = np.array(conf_matrix)
corrects = conf_matrix.diagonal(offset=0)
pre_kinds = conf_matrix.sum(axis=1)
print("混淆矩阵总元素个数：{0}，测试集总个数{1}".format(int(np.sum(conf_matrix)),len(f_p)))
print(conf_matrix)
print("每种故障总个数：", pre_kinds)
print("每种故障的识别准确率为：{0}".format([rate*100] for rate in corrects/pre_kinds))

fault_type = 8
# labels = ['cf','eo','fwc','fwe','nc','rl','ro','normal' ]
labels = ['F1','F2','F3','F4','F5','F6','F7','F0' ]
# plt.imshow(conf_matrix,cmap=plt.cm.Blues)
# thresh = conf_matrix.max()/2
# for x in range(fault_type):
#     for y in range(fault_type):
#         info = round(conf_matrix[y,x]/len(f_p)*100,2)
#         plt.text(x,y,info,
#                  verticalalignment='center',
#                  horizontalalignment='center',color="white" if info>thresh else "black")
normalized_conf_matrix = normalize(conf_matrix,axis=1,norm='l1')*100
sns.heatmap(normalized_conf_matrix, annot=True,fmt='.2f', cmap='Blues', cbar=True, square=True,annot_kws={'size':12})
plt.tight_layout()
plt.yticks(range(fault_type),labels,fontsize=12)
# plt.xticks(range(fault_type),labels,rotation=45,fontsize=12)
plt.xticks(range(fault_type),labels,fontsize=12)
plt.show()

# showdata = testdata[:2000,:]
# # showlabel = testlabel[:2000]
# # showdata,_ = netD(showdata)
# #
# # tsne = TSNE(n_components=3, init='pca',perplexity=30,early_exaggeration=4,learning_rate=1000,n_iter=3000, verbose=1, random_state=33)
# # result = tsne.fit_transform(showdata.detach())
# # scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# # result = scaler.fit_transform(result)
# #
# # fig = plt.figure(figsize=(20, 20))
# # ax = fig.add_subplot(projection='3d')
# # # ax = fig.add_subplot()
# # ax.set_title('t-SNE process')
# # ax.scatter(result[:,0], result[:,1],result[:,2], c=showlabel, s=10)
# # # ax.scatter(result[:,0], result[:,1], c=showlabel, s=10)
# # plt.show()
# showdata = labeled_data
# showlabel = labels
# showdata,_ = netD(showdata)
# tsne = TSNE(n_components=2)
# result = tsne.fit_transform(showdata.detach())
# scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# # result = scaler.fit_transform(result)
# result = scaler.fit_transform(result)
# class_num = len(np.unique(labels))
# df = pd.DataFrame()
# df["y"] = showlabel
# df["comp1"] = result[:, 0]
# df["comp2"] = result[:, 1]
# # df["comp3"] = result[:, 2]
# hex = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink']

# data_label = []
# for v in df.y.tolist():
#     if v == 0:
#         data_label.append("c0")
#     elif v == 1:
#         data_label.append("c1")
#     elif v == 2:
#         data_label.append("c2")
#     elif v == 3:
#         data_label.append("c3")
#     elif v == 4:
#         data_label.append("c4")
#     elif v == 5:
#         data_label.append("c5")
#     elif v == 6:
#         data_label.append("c6")
#     elif v == 7:
#         data_label.append("c7")
# df["value"] = data_label
# # sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), z=df.comp3.tolist(), hue=df.value.tolist(), style=df.value.tolist(),
# #                 palette=sns.color_palette(hex, class_num),
# #                 markers={"c0": ".", "c1": ".", "c2": ".", "c3": ".", "c4": ".", "c5": ".", "c6": ".", "c7": "."},
# #                 # s = 10,
# #                 data=df).set(title="")  # T-SNE projection
# sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), z=df.comp2.tolist(),  hue=df.value.tolist(), style=df.value.tolist(),
#                 palette=sns.color_palette(hex, class_num),
#                 markers={"c0": ".", "c1": ".", "c2": ".", "c3": ".", "c4": ".", "c5": ".", "c6": ".", "c7": "."},
#                 # s = 10,
#                 data=df).set(title="")  # T-SNE projection
#
# # 指定图注的位置 "lower right"
# plt_sne.legend(loc = "lower right")
# plt_sne.axis("off")
# plt_sne.savefig(os.path.join("C:/Users\HNU\Desktop\对比\T-SNE\pic"",%s.jpg") % str(3),format = 'jpg', dpi = 300)
# fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(projection='3d')
# # ax = fig.add_subplot()
# ax.set_title('t-SNE process')
# # ax.scatter(result[:,0], result[:,1],result[:,2], c=showlabel, s=10)
# ax.scatter(result[:,0], result[:,1], c=showlabel, s=10)
# plt.show()


