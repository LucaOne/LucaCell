

# LucaCell  
     

## 环境安装             

### step1: update git                       

#### 1) centos              
sudo yum update     
sudo yum install git-all

#### 2) ubuntu              
sudo apt-get update     
sudo apt install git-all

### step2: install python 3.9               

#### 1) download anaconda3             
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh           

#### 2) install conda         
sh Anaconda3-2022.05-Linux-x86_64.sh              

##### Notice: Select Yes to update ~/.bashrc            
source ~/.bashrc             

#### 3) create a virtual environment: python=3.9.13              
conda create -n lucacell python=3.9.13         


#### 4) activate lucacell           
conda activate lucacell  
        


### step3:  install other requirements          
conda create -n lucacellv2 python=3.9.13             

conda activate lucacellv2        

pip install torch==2.5.1 torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121           

pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple           

pip install flash-attn --no-build-isolation  -i https://pypi.tuna.tsinghua.edu.cn/simple         

pip install -r requirement_others.txt -i https://pypi.tuna.tsinghua.edu.cn/simple          


## pre-training   
`/src/training/multi/run_multi_v2.0_8_vector_1200_final.sh`


