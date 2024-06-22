from pathlib import Path
import random

import clip
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchmetrics
import torchvision
from tqdm import tqdm

from balanced_batch_sampler import BalancedBatchSampler
import os
from collections import defaultdict
from PIL import Image
import gc
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
SAVE_INTERVAL = 10
BATCH_SIZE = 8
NUM_EPOCHS = 30
def convert_models_to_fp32(model):
    """cpu不支持半浮点数,所以需要转换"""
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  #Must set jit=False for training
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
class LimitedImageFolder(Dataset):
    def __init__(self, root, transform=None, limit_per_class=4):
        self.root = root
        self.transform = transform
        self.limit_per_class = limit_per_class
        self.samples = self._gather_samples()
        self.targets = [s[1] for s in self.samples]

    def _gather_samples(self):
        samples = []
        class_counts = defaultdict(int)  # 初始化类别计数字典

        # 遍历每个数据集文件夹
        for dataset_dir in os.listdir(self.root):
            dataset_path = os.path.join(self.root, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            # 遍历每个类别文件夹
            for class_dir in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_dir)
                if not os.path.isdir(class_path):
                    continue

                # 获取类别内的所有图像路径
                class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
                
                # 随机选择limit_per_class张图像
                selected_images = random.sample(class_images, min(self.limit_per_class, len(class_images)))
                for img_path in selected_images:
                    class_idx = class_dir
                    samples.append((img_path, class_idx))
                    class_counts[class_idx] += 1  # 更新类别计数

        return samples

    def __len__(self):
        # print("len是",len(self.samples))
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target
train_dataset=LimitedImageFolder("data/TrainSet", transform=preprocess,limit_per_class=4)
train_dataloader = torch.utils.data.DataLoader(train_dataset,drop_last=False,shuffle=True,num_workers=4,batch_size=BATCH_SIZE)
test_dataset=LimitedImageFolder("data/TrainSet", transform=preprocess,limit_per_class=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset,drop_last=True,shuffle=True,num_workers=4,batch_size=BATCH_SIZE)
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
for p in model.transformer.parameters():
    p.requires_grad = True
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-7, weight_decay=0.0001)
num_batches_train = len(train_dataloader.dataset)/BATCH_SIZE
writer = SummaryWriter()
weights_path = Path("model_checkpoints")
weights_path.mkdir(exist_ok=True)
def ret_class_name_dic()->dict:
    """返回动物名字到数字的字典"""
    classes = open('data/classname.txt').read().splitlines()#这是一个包含所有类的列表
    class_name_dic={}#这是数字映射到动物名字的字典
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
        if c not in class_name_dic:
            class_name_dic[c]=idx
        else:
            print(name,"already exist!!")
    return class_name_dic
class_dic=ret_class_name_dic()
class_list=list(class_dic.keys())
def acc_compu(pre,true_label):
    count_acc=0
    count_total=0
    for i in range(len(pre)):
        # print(int(pre[i]))
        # print("@@@@",true_label[i].tolist())

        if int(pre[i]) in true_label[i].tolist():
            count_acc+=1
            # print("Ok")
        count_total+=1
        
    
    return count_acc/count_total

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    epoch_train_loss = 0
    model.train()
    for batch in tqdm(train_dataloader,total=num_batches_train):
        # break
        optimizer.zero_grad()

        images, class_ids = batch

        images = torch.stack([img for img in images], dim=0).to(
            device
        )
        # TODO: to use mean of multiple prompts need to pre-compute them.
        texts = [f"a photo of a {label_id}" for label_id in class_ids]
        texts = clip.tokenize(texts).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)
        # print("#####logits_per_image.shape[0]是",logits_per_image.shape[0])
        # print(ground_truth,"#####ground_truth.shape是",logits_per_image.shape)
        total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_train_loss.backward()
        epoch_train_loss += total_train_loss

        torch.nn.utils.clip_grad_norm_(params, 1.0)

        optimizer.step()
        

    epoch_train_loss /= num_batches_train
    writer.add_scalar("Loss/train", epoch_train_loss, epoch)
    model.eval()
    values_list, indices_list = [], []
    top5_results = []
    top1_results = []
    acc_top1_list = []
    acc_top5_list = []

    all_test_logits = []
    all_test_labels = []
    epoch_test_loss = 0

    num_batches_test = len(test_dataloader.dataset)/BATCH_SIZE
    epoch_test_loss = 0
    cout=0
    for i, batch in enumerate(tqdm(test_dataloader, total=num_batches_test)):
        images, class_ids = batch
        # class_ids = class_ids.to(device)

        images = images.to(device)
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_list]).to(device)
        text2 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_ids]).to(device)
        with torch.no_grad():
            # TODO: remove duplicate computation of image and text features
            image_features = model.encode_image(images)
            text_features = model.encode_text(text2)

            logits_per_image, logits_per_text = model(images, text2)
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            epoch_test_loss += total_loss
            
            #计算准确率
            text_features = model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).to(device)
            true_label=torch.tensor([int(class_dic[i]) for i in class_ids]).to(device)
            # 记录测试集的logits和真实标签
            all_test_logits.append(similarity)
            all_test_labels.extend(class_ids)
    all_test_logits = torch.cat(all_test_logits).to(device)
    all_test_labels = torch.tensor([int(class_dic[label]) for label in all_test_labels]).to(device)
    aa=all_test_logits
    bb=all_test_labels
    
    acc=acc_compu(all_test_labels,all_test_logits.topk(1, dim=1).indices)

    print(f"Epoch {epoch} test acc: {acc}")
    
    gc.collect()
    torch.cuda.empty_cache()

# writer.flush()
writer.close()

# %%
# def acc_compu(pre,true_label):
#     count_acc=0
#     count_total=0
#     for i in range(len(pre)):
#         # print(int(pre[i]))
#         # print("@@@@",true_label[i].tolist())

#         if int(pre[i]) in true_label[i].tolist():
#             count_acc+=1
#             # print("Ok")
#         count_total+=1
        
    
#     return count_acc/count_total
# acc_compu(bb,aa.topk(1, dim=1).indices)

# %%

classes = open('data/classname.txt').read().splitlines()
model.eval()
# 冻结所有模型参数
for param in model.parameters():
    param.requires_grad = False
# remove the prefix Animal, Thu-dog, Caltech-101, Food-101
gc.collect()
torch.cuda.empty_cache()
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

text = clip.tokenize(new_classes).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

split = 'TestSetA' 

imgs_dir = 'data/' + split
imgs = os.listdir(imgs_dir)


# %%

save_file = open('result.txt', 'w')

preds = []
model.eval()
count=0
for img in tqdm(imgs):
    img_path = os.path.join(imgs_dir, img)
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 *
                    image_features @ text_features.transpose(0, 1)).softmax(
                        dim=-1)
    # top5 predictions
    _, top_labels = text_probs[0].topk(5)
    preds.append(top_labels)
    # save top5 predictions to file
    save_file.write(img + ' ' +
                    ' '.join([str(p.item()) for p in top_labels]) + '\n')
    del image,_, top_labels,image_features,img_path,text_probs
    gc.collect()
    torch.cuda.empty_cache()
    count+=1
print(count)
    

# %%



