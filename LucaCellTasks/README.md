Downstream Tasks of LucaCell

1. Networks      
Three distinct networks correspond to three different types of inputs:                 
lucasingle (Single)            
lucapair_heter (Heterogeneous Pair)         
lucapair_homo (Homogeneous Pair)           

Downstream task network with three input types        

Cross-species cell type annotation      
Input: cell gene expression profile (single cell)           
Network: lucasingle (src/lucasingle/models/LucaSingle.py)           

Alignment-free cell embedding            
Input: cell RNA-seq reads (single cell)           
Network: lucasingle (src/lucasingle/models/LucaSingle.py)        

Gene perturbation prediction        
Input: cell gene expression profile + gene sequences (heterogeneous pair)         
Network: lucapair_heter (src/lucapair/models/LucaPairHeter.py)           

SNP-aware gene expression modeling       
Input: cell gene expression profile carrying SNP information + gene sequences carrying SNP information (heterogeneous pair)        
Network: lucapair_heter (src/lucapair/models/LucaPairHeter.py)           

Viral load prediction        
Input: cell gene expression profile + virus genome (homogeneous pair)        
Network: lucapair_homo (src/lucapair/models/LucaPairHomo.py)            

2. Installation      
1) create a virtual environment: python=3.9.13        
conda create -n lucacell_tasks python=3.9.13         

2) activate lucacell_tasks       
conda activate lucacell_tasks         

3) install other requirements         
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple       

3. Datasets     
Copy the datasets from https://zenodo.org/records/22073984/ into the directory ./dataset/        

4. LucaCell Trained Checkpoint        
Copy the checkpoint files from https://zenodo.org/records/22073984/ into the directory ./model         

5. Usage of Downstream Models Inference          
Use the script src/prediction.sh to load the trained model and predict.         

6. Downstream Tasks         
The running scripts of downstream tasks:       
1) src/training/lucacell/run_task1_cellxgene_kidney_rmMouse1_cell_type_lucasinglev2.init_weight.sh : cross-species cell type annotation.        
2) src/training/lucacell/run_task1_smRandom_CRA016741_lucasinglev2.init_weight.sh : alignment-free microbe embedding          
3) src/training/lucacell/run_task3_GSE90546_pertub_heter_lucapair_heterv2.sh: gene perturbation            
4) src/training/lucacell/run_task3_cls_onek1k_nosnp_heter_lucapair_heter.sh: gene expression modeling without SNP information.          
5) src/training/lucacell/run_task3_cls_onek1k_snp_heter_lucapair_heter.sh: SNP-informed gene expression modeling.            
6) src/training/lucacell/run_task3_bronchoalveolar_load_lucapair_homo.lucacellv2.sh: viral load prediction.           

