# PyTorch-Study

## 入门
去看看莫凡教程吧：https://morvanzhou.github.io/

## 安装环境 - 1080Ti

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

7. 禁用 lightdm 桌面服务: sudo service lightdm stop (关键！一定要执行！)

8. 禁用 nouveau 显卡驱动: sudo vim /etc/modprobe.d/blacklist.conf
```
# 添加
sudo vim /etc/modprobe.d/blacklist.conf

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

### 安装 NVIDIA
1. 权限
```
sudo chmod a+x NVIDIA-Linux-x86_64-390.20.run
```

2. 安装
```
sudo sh NVIDIA-Linux-x86_64-390.25.run -no-x-check -no-nouveau-check -no-opengl-files
# 询问'Would you like to run the nvidia-xconfig utility......'，选择N
# -no-x-check # 安装驱动时关闭 X 服务
# -no-nouveau-check # 安装驱动时禁用 nouveau 驱动
# -no-opengl-files # 只安装驱动文件，不安装 opengl 文件
```

### 安装 CUDA
1. 权限
```
sudo chmod a+x cuda_9.0.176_384.81_linux.run
```

2. 安装
```
sudo sh cuda_9.0.176_384.81_linux.run
# 询问'Install NVIDIA Accelerated Graphics Driver for......'，选择N
```

3. 添加环境变量 vim ~/.bashrc
```
export PATH=$PATH:/usr/local/cuda-9.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
```

### 安装 cuDNN
1. 解压
```
tar -zxvf cudnn-9.0-linux-x64-v7.tgz
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

### 安装 Anaconda
https://www.anaconda.com/
1. 权限
```
sudo chmod a+x Anaconda3-5.0.1-Linux-x86_64.sh
```

2. 安装
```
./Anaconda3-5.0.1-Linux-x86_64.sh
# 询问安装路径，默认回车
# 询问环境变量，选择yes
```

3. 重新加载环境变量
```
source ~/.bashrc
```

### 安装 PyTorch
https://pytorch.org/
1. pip方式
```
pip install torch torchvision
```

2. conda方式
```
conda install pytorch torchvision -c pytorch
```

### 安装 Jupyter
1. 安装 Jupyter:
```
pip install jupyter
```

2. 安装 OpenSSH:
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

4. 生成配置文件:
```
jupyter notebook --generate-config
```

5. 修改配置文件:vim ~/.jupyter/jupyter\_notebook\_config.py
```
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:......加密密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

6. 启动 Jupyter:
```
jupyter notebook
```

7. 远程访问:
```
ssh username@address_of_remote -L localhost:1234:localhost:8888
```

8. 浏览器访问: http://localhost:1234

### 安装 Samba
1. 安装 Samba
```
sudo apt-get install samba
```

2. 修改配置文件 sudo vim /etc/samba/smb.conf
```
[share]
comment = Shared Folder for zzz
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

3. 创建目录
```
mkdir share
```

4. 赋予权限
```
chmod 777 share
```

5. 设置密码
```
sudo smbpasswd -a ubuntu

New SMB password: xxxxxx
Retype new SMB password: xxxxxx
```

6. 重启服务
```
sudo /etc/init.d/samba restart
```

7. 共享目录：打开 \\\\IP地址（linux）\share
