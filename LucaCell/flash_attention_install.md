# flash attention    
conda create -n lucacell_flash_attention python=3.9.13     
conda activate lucacell_flash_attention     

pip install torch==2.5.1 torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
# 下面一行不一定需要
pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple     
pip install flash-attn --no-build-isolation  -i https://pypi.tuna.tsinghua.edu.cn/simple     
or  
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --no-cache-dir     
pip install -r requirement_others.txt -i https://pypi.tuna.tsinghua.edu.cn/simple      