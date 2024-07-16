# PLM_Sol

What is PLM_Sol?
=============
A protein solubility prediction tool based on protT5.

Env
=============
protT5 environment https://github.com/HannesStark/protein-localization.
```
# Pytorch==2.0.1 CUDA Version: 11.4 
conda env create -f env.yml
conda activate PLM_Sol
pip install -r requirements.txt
```
Using PLM_Sol
=============

Used the bio-embedding to generate the .h5 file
```
cd embedding_datset
#Change the 
bio_embeddings light_attention_protT5.yml
```
Training
```
python train.py --config ./configs/SOL_biLSTM_TextCNN.yml
```
Predict
```
python inference.py --config ./configs/inference_Sol_biLSTM_TextCNN.yml

```

