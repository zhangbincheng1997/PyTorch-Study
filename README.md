# PyTorch-Study

## 安装环境

### [更新源](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/)
Ubuntu 的软件源配置文件是 /etc/apt/sources.list。将系统自带的该文件做个备份，将该文件替换为下面内容，即可使用 TUNA 的软件源镜像。
```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```
### 安装 NVIDIA
1. 添加源
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
```

2. 推荐驱动
```
$ sudo ubuntu-drivers devices

driver   : nvidia-driver-418 - third-party free recommended
```

3. 安装驱动
```
$ sudo apt-get install nvidia-driver-418

ps: 卸载驱动 sudo apt-get remove --purge nvidia-*
```

### 安装 CUDA
1. 下载 CUDA10.0:
https://developer.nvidia.com/cuda-downloads
Linux - x86_64 - Ubuntu - 16.04 - runfile(local)
> * https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux

2. 安装
```
$ chmod a+x cuda_10.0.130_410.48_linux.run
$ sudo ./cuda_10.0.130_410.48_linux.run
$ vim ~/.bashrc

export PATH=$PATH:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
```

### 安装 cuDNN
1. 下载 cuDNN10.0:
https://developer.nvidia.com/rdp/cudnn-download
Download cuDNN v7.5.0 for CUDA10.0 - cuDNN Library for Linux
> * https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.0.56/prod/10.0_20190219/cudnn-10.0-linux-x64-v7.5.0.56.tgz

2. 安装
```
$ tar -zxvf cudnn-10.0-linux-x64-v7.5.0.56.tgz
$ cd cuda
$ sudo cp lib64/* /usr/local/cuda/lib64/
$ sudo cp include/* /usr/local/cuda/include/
```

### 检测环境
```
$ sudo reboot # 重启
$ nvidia-smi  # 检测NVIDIA
$ nvcc -V     # 检测CUDA
```

### 安装 Anaconda
1. 下载 Anaconda3.5:
> * https://repo.anaconda.com/archive/Anaconda3-5.0.0-Linux-x86_64.sh
> * https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.0-Linux-x86_64.sh

2. . 安装
```
$ chmod a+x Anaconda3-5.0.0-Linux-x86_64.sh
$ ./Anaconda3-5.0.1-Linux-x86_64.sh
$ source ~/.bashrc
```

### 安装 PyTorch
1. pip方式
```
$ pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
$ pip install torchvision
```

2. conda方式
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

### 安装 Jupyter
1. 安装 Jupyter
```
$ pip install jupyter
```

2. 安装 OpenSSH
```
$ sudo apt-get install openssh-server
```

3. 生成加密密文
```
$ python

In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:......加密密文'
```

4. 生成配置文件
```
$ jupyter notebook --generate-config
```

5. 修改配置文件
```
$ vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:......加密密文'
c.NotebookApp.port = 8888
```

6. 启动 Jupyter
```
$ jupyter notebook
```

7. 端口转发
```
$ ssh username@server_ip -L localhost:1234:localhost:8888
```

8. 远程访问: http://localhost:1234

### 安装 Samba
1. 安装 Samba
```
$ sudo apt-get install samba
```

2. 修改配置文件
```
$ sudo vim /etc/samba/smb.conf

[share]
comment = Shared Folder
path = /home/share
public = yes
writable = yes
valid users = ubuntu
create mask = 0755
directory mask = 0755
force user = nobody
force group = nogroup
available = yes
browseable = yes
```

3. 设置登录密码
```
$ sudo smbpasswd -a username
```

4. 重启服务
```
$ sudo samba restart
```

5. 打开共享目录: smb://server_ip/share/
