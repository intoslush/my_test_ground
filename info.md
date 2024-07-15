## 换源
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple

## cmd走代理
set https_proxy=http://127.0.0.1:7890

## 下载比赛数据
wget https://cloud.tsinghua.edu.cn/f/212edd1e7b3b44f5b301/?dl=1 -O train.txt
wget https://cloud.tsinghua.edu.cn/f/418b311c5ae8484f8208/?dl=1 -O classname.txt
wget https://cloud.tsinghua.edu.cn/f/7c44b138a6344f4b8fd1/?dl=1 -O TrainSet.zip
wget https://cloud.tsinghua.edu.cn/f/c00ca0f3f27340899a05/?dl=1 -O TestSetA.zip
mocov3的预训练模型
wget https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar

## 下载官方给的bassline的训练参数
wget https://github.com/uyzhang/JCLIP/releases/download/%E6%9D%83%E9%87%8D/ViT-B-32.pkl -O ViT-B-32.pkl


## 克隆base 环境
conda create --name pyt --clone base

## 配置git名字和邮箱
git config --global user.name "ZA"
git config --global user.email "intoslush@gmail.com"
## git相关命令 
git reset HEAD
git reset HEAD~1

## 本项目克隆地址
https://github.com/intoslush/my_test_ground.git
git gc --aggressive --prune=all
git push origin master --force

## amu测试
下载牛津宠物数据集
pip install gdown
#Create a folder named oxford_pets under $DATA
export DATA=data/
mkdir -p $DATA/oxford_pets

#Download the images
wget -P $DATA/oxford_pets https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

#Download the annotations
wget -P $DATA/oxford_pets https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

#Download split_zhou_OxfordPets.json from Google Drive
gdown --output $DATA/oxford_pets/split_zhou_OxfordPets.json "https://drive.google.com/uc?id=1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs"
#Extract the images
tar -xzvf $DATA/oxford_pets/images.tar.gz -C $DATA/oxford_pets

#Extract the annotations
tar -xzvf $DATA/oxford_pets/annotations.tar.gz -C $DATA/oxford_pets

#Delete the tar.gz files after extraction
rm $DATA/oxford_pets/images.tar.gz
rm $DATA/oxford_pets/annotations.tar.gz
## 必要的包
pip install ftfy regex pyyaml
pip install scipy