# %%
# %pip install ftfy regex tqdm
# %pip install colorama

# %%
import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
import numpy as np
from colorama import Fore, Back, Style, init
init()

jt.flags.use_cuda = 1
print(Back.YELLOW +"包导入成功")

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
model, preprocess = clip.load("ViT-B-32.pkl")
classes = open('../data/classname.txt').read().splitlines()#这是一个包含所有类的列表

# %% [markdown]
# ## 类别文本编码,变成一句话,然后别转为向量
# ### 提示词可以尝试优化
# text_features的shape为[374,512,]

# %%
# encode这块后面可以强化下
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
    text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
print(Back.YELLOW +"标签处理成功")

# %% [markdown]
# ## 图片加载并处理

# %%
# training data loading
imgs_dir = '../data'
train_labels = open('../data/train.txt').read().splitlines()
train_imgs = [l.split(' ')[0] for l in train_labels]#对应的图片path
train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]#对应的种类序号

# %%
# 每个类挑四张图，根据train_labels中的label来挑选
#挑选每种的前四张,生成两个对应的列表,分别存储path和类别信息
cnt = {}
new_train_imgs = []
new_train_labels = []
for i in range(len(train_imgs)):
    label = int(train_labels[i].numpy())
    if label not in cnt:
        cnt[label] = 0
    if cnt[label] < 4:
        new_train_imgs.append(train_imgs[i])
        new_train_labels.append(train_labels[i])
        cnt[label] += 1

# %%
# calculate image features of training data
train_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        print("能成功运行?","image_features的shape是",image_features.shape)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        train_features.append(image_features)

train_features = jt.cat(train_features).numpy()
train_labels = jt.cat(new_train_labels).numpy()

# %%
train_features[0]

# %%



