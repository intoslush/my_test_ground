
from pathlib import Path
import random

import clip
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

import torchmetrics
import torchvision
from tqdm import tqdm

from back_save.为了测试clip的原始算法.balanced_batch_sampler import BalancedBatchSampler
import os
from collections import defaultdict
from PIL import Image


EPOCH=30
BATCH_SIZE=16

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt,nidx):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        # self.title=self.title
        self.nidx=nidx

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        ndix=self.nidx[idx]
        return image,title,ndix


def ret_class_name_dic()->dict:
    """返回动物名字到数字和数字映射到动物名的字典"""
    classes = open('data/classname.txt').read().splitlines()#这是一个包含所有类的列表
    class_name_dic_num={}
    class_name_dic_name={}
    for i in classes:
        name,idx = i.split(' ')
        c = name
        if c.startswith('Animal'):
            c = c[7:]
        if c.startswith('Thu-dog'):
            c = c[8:]
        if c.startswith('Caltech-101'):
            c = c[12:]
        if c.startswith('Food-101'):
            c = c[9:]
        if c not in class_name_dic_name:
            class_name_dic_name[c]=idx
            class_name_dic_num[idx]=c
        else:
            print(name,"already exist!!")
    return class_name_dic_name,class_name_dic_num
class_name_dic_name,class_name_dic_num=ret_class_name_dic()

def ret_pic_patch()->dict:
    """返回每类四张,的路径和标签"""
    r_path=[]
    r_class_num=[]
    info = open('data/train.txt').read().splitlines()
    
    class_check=0
    temp_path=[]
    temp_class=[]
    for i in info:
        path,class_num=i.split(' ')
        path="data/"+path
        if class_check==int(class_num):
            temp_path.append(path)
            temp_class.append(class_num)
        else:
            class_check=int(class_num)

            r_path+=random.sample(temp_path,4)
            r_class_num+=random.sample(temp_class,4)
            temp_path=[path]
            temp_class=[class_num]
    r_path+=random.sample(temp_path,4)
    r_class_num+=random.sample(temp_class,4)
    return r_path,r_class_num

path,class_num=ret_pic_patch()
print(path[0:5],"\n",class_num[0:5])
print(class_name_dic_num['0'])
list_text_raw=["a photo of a "+class_name_dic_num[i] for i in class_num]
print(list_text_raw[0:5])
print(len(list_text_raw),len(class_name_dic_name.keys()))

list_image_path = path
list_txt = list_text_raw

dataset = image_title_dataset(list_image_path,list_txt,class_num)
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=True)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

classes = open('data/classname.txt').read().splitlines()
new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

print(new_classes[0:5])
# 字符编码每一类
text2 = clip.tokenize(new_classes).to(device)

for epoch in range(EPOCH):
    total_count=0
    total_count1=0
    total_count5=0
    count_loss=0
    model.train()
    for batch in train_dataloader :
        optimizer.zero_grad()

        images,texts,idx = batch 
        images= images.to(device)
        texts = texts.to(device)
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        # print("total_loss是:",total_loss)
        count_loss+=total_loss
        total_loss.backward()
        optimizer.step()
        
    model.eval()
    for batch in train_dataloader :
        images,texts,idx = batch 
        images= images.to(device)
        texts = texts.to(device)
        logits_per_image, logits_per_text =model(images, text2)
        text_probs=logits_per_image.softmax(dim=-1)
        for i in range(len(idx)):
            top5=text_probs[i].topk(5).indices.tolist()

            if int(idx[i]) in top5:
                total_count5+=1
                if int(idx[i])==top5[0]:
                    total_count1+=1
        total_count+=BATCH_SIZE

        
    print("训练集准确率top1",total_count1/total_count,"\ntop5有",total_count5/total_count)
    print('loss为',count_loss)
    del count_loss,total_count1,total_count,total_count5
    
      


# %%
# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss,
#         }, f"model_checkpoint/model_10.pt") #just change to your preferred folder/filename

# # 加载之前训练的模型
# model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# checkpoint = torch.load("model_checkpoint/model_10.pt")

# # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

# model.load_state_dict(checkpoint['model_state_dict'])


