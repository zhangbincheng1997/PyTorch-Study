# PyTorch-Study

## 安装环境 - 1080Ti

### 准备工作
1. 下载 NVIDIA418:
http://www.nvidia.cn/Download/index.aspx
GeForce - GeForce 10 Series - GeForce GTX 1080 Ti - Linux 64-bit - Chinese(Simplified)
> * https://www.nvidia.cn/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/418.43/NVIDIA-Linux-x86_64-418.43.run&lang=cn&type=TITAN

2. 下载 CUDA10.0:
https://developer.nvidia.com/cuda-downloads
Linux - x86_64 - Ubuntu - 16.04 - runfile(local)
> * https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux

3. 下载 cuDNN10.0:
https://developer.nvidia.com/rdp/cudnn-download
Download cuDNN v7.5.0 for CUDA10.0 - cuDNN Library for Linux
> * https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.0.56/prod/10.0_20190219/cudnn-10.0-linux-x64-v7.5.0.56.tgz

4. 下载 Anaconda3.5:
> * https://repo.anaconda.com/archive/Anaconda3-5.0.0-Linux-x86_64.sh
> * https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.0-Linux-x86_64.sh

5. 关闭 BIOS 安全启动

6. 进入命令行模式: ctrl + alt + F1 (关键)

7. 禁用 lightdm 桌面服务
```
sudo service lightdm stop
```

8. 禁用 nouveau 显卡驱动
```
# 1.添加黑名单
sudo vim /etc/modprobe.d/blacklist.conf

blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb

# 2. 更新内核
sudo update-initramfs -u

# 3. 重启系统
sudo reboot

# 4. 检查屏蔽
lsmod | grep nouveau
```

### 安装 NVIDIA
1. 权限
```
sudo chmod a+x NVIDIA-Linux-x86_64-418.43.run
```

2. 安装
```
sudo sh NVIDIA-Linux-x86_64-418.43.run -no-opengl-files
# -no-opengl-files # 只安装驱动文件，不安装 opengl 文件（核显输出，独显运算）
```

### 安装 CUDA
1. 权限
```
sudo chmod a+x cuda_10.0.130_410.48_linux.run
```

2. 安装
```
sudo sh cuda_10.0.130_410.48_linux.run
# 询问'Install NVIDIA Accelerated Graphics Driver for......'，选择N
```

3. 添加环境变量
```
vim ~/.bashrc

export PATH=$PATH:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
```

### 安装 cuDNN
1. 解压
```
tar -zxvf cudnn-10.0-linux-x64-v7.5.0.56.tgz
```

2. 链接
```
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
```

### 检测环境
```
sudo reboot # 重启
nvidia-smi  # 检测驱动
nvcc -V     # 检测CUDA
```

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

### 安装 Anaconda
1. 权限
```
sudo chmod a+x Anaconda3-5.0.0-Linux-x86_64.sh
```

2. 安装
```
./Anaconda3-5.0.1-Linux-x86_64.sh
# 询问安装路径，默认回车
# 询问环境变量，选择yes
```

3. 强制更新环境变量
```
source ~/.bashrc
```

### 安装 PyTorch
1. pip方式
```
pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```

2. conda方式
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

### 安装 Jupyter
1. 安装 Jupyter
```
pip install jupyter
```

2. 安装 OpenSSH
```
sudo apt-get install openssh-server
```

3. 生成加密密文
```
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:......加密密文'
```

4. 生成配置文件
```
jupyter notebook --generate-config
```

5. 修改配置文件
```
vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:......加密密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

6. 启动 Jupyter
```
jupyter notebook
```

7. 端口转发
```
ssh username@server_ip -L localhost:1234:localhost:8888
```

8. 远程访问
```
http://localhost:1234
```

### 安装 Samba
1. 安装 Samba
```
sudo apt-get install samba
```

2. 修改配置文件
```
sudo vim /etc/samba/smb.conf

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
sudo smbpasswd -a username

New SMB password: xxxxxx
Retype new SMB password: xxxxxx
```

4. 重启服务
```
sudo samba restart
```

5. 打开共享目录
```
smb://server_ip/share/
```
