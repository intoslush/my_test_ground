
## 换源
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
Writing to C:\Users\12939\AppData\Roaming\pip\pip.ini
## cmd走代理
set https_proxy=http://127.0.0.1:7890
## 下载比赛数据
wget https://cloud.tsinghua.edu.cn/f/212edd1e7b3b44f5b301/?dl=1 -O train.txt
wget https://cloud.tsinghua.edu.cn/f/418b311c5ae8484f8208/?dl=1 -O classname.txt
wget https://cloud.tsinghua.edu.cn/f/7c44b138a6344f4b8fd1/?dl=1 -O TrainSet.zip

wget https://cloud.tsinghua.edu.cn/f/c00ca0f3f27340899a05/?dl=1 -O TestSetA.zip
## 下载官方给的bassline的训练参数
wget https://github.com/uyzhang/JCLIP/releases/download/%E6%9D%83%E9%87%8D/ViT-B-32.pkl -O ViT-B-32.pkl

## 运行demo时出现bug修改了如下参数
好像是显存用完了
export JT_SYNC=1
export trace_py_var=3
## 克隆base 环境
conda create --name pyt --clone base
## 配置git名字和邮箱
git config --global user.name "ZA"
git config --global user.email "intoslush@gmail.com"
## 克隆地址
https://github.com/intoslush/my_test_ground.git