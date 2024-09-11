# PLM_Sol

What is PLM_Sol?
=============
A protein solubility prediction tool based on protT5.
![image](https://github.com/Violet969/PLM_Sol/blob/main/PLM_Sol_arch.png)

Env
=============
protT5 environment https://github.com/HannesStark/protein-localization.
```
# Pytorch==2.0.1 CUDA Version: 11.4 
conda env create -f env.yml
conda activate PLM_Sol
pip install -r requirements.txt
pip install bio-embeddings[all]
```
Using PLM_Sol
=============

Used the bio-embedding to generate the .h5 file
```
cd embedding_datset
#Change the file path (sequences_file: ./Train_dataset.fasta prefix: ./Train_dataset_emb)
bio_embeddings embedding_protT5.yml
```
Training
```
#Change the file path of .h5 and .fasta
python train.py --config ./configs/SOL_biLSTM_TextCNN.yml
```
Predict
```
#Change the file path of .h5 and .fasta
python inference.py --config ./configs/inference_Sol_biLSTM_TextCNN.yml
```
Then you can use the PLM_Sol_csv.ipynb to merge the orignal file and predicted csv file.

Cite PLM_Sol
=============
```
@article{zhang2024plm_sol,
  title={PLM\_Sol: predicting protein solubility by benchmarking multiple protein language models with the updated Escherichia coli protein solubility dataset},
  author={Zhang, Xuechun and Hu, Xiaoxuan and Zhang, Tongtong and Yang, Ling and Liu, Chunhong and Xu, Ning and Wang, Haoyi and Sun, Wen},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={5},
  pages={bbae404},
  year={2024},
  publisher={Oxford University Press}
}
```
