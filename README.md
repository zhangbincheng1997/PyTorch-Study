# PyTorch-Study

## 入门
> * 0_autograd.py 自动梯度
> * 1_activation.py 激励函数
> * 2_regression.py 回归
> * 3_classification.py 分类
> * 4_reload.py 保存提取
> * 5_batch.py 批训练
> * 6_optimizer.py 优化器

## 高阶
download it if you don't have it  
set DOWNLOAD_XXXX = True  
> * autoencoder/autoencoder.py 自编码
> * cnn/cnn.py 卷积神经网络
> * rnn/rnn.py 循环神经网络
> * gan/mnist_gan.py 生成对抗网络 GAN
> * gan/mnist_dcgan.py 生成对抗网络 DCGAN

## 安装环境 - 1080Ti

### Ubuntu 16.04
1. 安装谷歌浏览器: https://www.google.cn/chrome/browser/desktop/
```
sudo dpkg -i xxx
sudo apt-get -f install
```
2. 安装搜狗输入法: https://pinyin.sogou.com/linux/
```
sudo dpkg -i xxx
sudo apt-get -f install
```
3. 设置屏幕分辨率
```
vim .profile
# 添加
xrandr --newmode "1920x1080_60.00" 173.00 1920 2048 2248 2576 1080 1083 1088 1120 -hsync +vsync
xrandr --addmode VGA-1 "1920x1080_60.00"
```

### 准备工作
1. 下载 NVIDIA390:
http://www.nvidia.cn/Download/index.aspx
GeForce - GeForce 10 Series - GeForce GTX 1080 Ti - Linux 64-bit - Chinese(Simplified)

2. 下载 CUDA9.0:
https://developer.nvidia.com/cuda-90-download-archive
Linux - x86_64 - Ubuntu - 16.04 - runfile(local)

3. 下载 cuDNN7.0:
https://developer.nvidia.com/rdp/cudnn-download
cuDNN v7.0.5 Library for Linux

4. 下载:
Anaconda3.6 https://www.anaconda.com/download/#linux
Python 3.6 version * - Download

5. 关闭 BIOS 安全启动

6. 进入命令行模式: ctrl + alt + F1 (关键)

7. 禁用 lightdm 桌面服务: sudo service lightdm stop (关键)

8. 禁用 nouveau 显卡驱动: 
```
sudo vim /etc/modprobe.d/blacklist.conf
# 添加
blacklist vga16fb 
blacklist nouveau 
blacklist rivafb 
blacklist rivatv 
blacklist nvidiafb
# 更新内核
sudo update-initramfs -u
# 重启系统
sudo reboot
# 检查屏蔽
lsmod | grep nouveau
```

### 安装 NVIDIA 驱动
```
# 权限
sudo chmod a+x NVIDIA-Linux-x86_64-390.20.run
# 安装
sudo sh NVIDIA-Linux-x86_64-390.25.run -no-x-check -no-nouveau-check -no-opengl-files
# -no-x-check # 安装驱动时关闭 X 服务
# -no-nouveau-check # 安装驱动时禁用 nouveau 驱动
# -no-opengl-files # 只安装驱动文件，不安装 opengl 文件
```

### 安装 CUDA 工具包
```
# 权限
sudo chmod a+x NVIDIA-Linux-x86_64-390.20.run
# 安装
sudo sh cuda_9.0.176_384.81_linux.run
# 询问是否安装附带驱动，选择N
vim ~/.bashrc
# 添加
export PATH=$PATH:/usr/local/cuda-9.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
```

### 安装 cuDNN 框架
```
# 解压
tar -zxvf cudnn-9.0-linux-x64-v7.tgz
cd cuda
# 链接
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
```

### 检测环境
```
# 重启
sudo reboot
# 驱动
nvidia-smi
# CUDA
nvcc -V
```

### 安装 Anaconda 工具包
```
# 默认
./Anaconda3-5.0.1-Linux-x86_64.sh
# 同步
source ~/.bashrc
```

### 安装深度学习框架
1. TensorFlow:
```
pip install tensorflow-gpu
```

2. Keras:
```
pip install keras
```

3. PyTorch:
```
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```

### 安装交互式笔记本
1. 安装 Jupyter:
pip install jupyter

2. 安装远程登录:
sudo apt-get install openssh-server

3. 生成加密密文
```
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:......'
```

4. 生成配置文件:
jupyter notebook --generate-config

5. 修改配置文件:
vim ~/.jupyter/jupyter_notebook_config.py 
```
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:...加密密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

6. 启动 Jupyter:
jupyter notebook

7. 远程访问:
ssh username@address_of_remote -L localhost:1234:localhost:8888

8. 浏览器访问: localhost:1234
