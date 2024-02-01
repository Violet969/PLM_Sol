# PLM_Sol

What is PLM_Sol?
=============
A protein solubility prediction tool based on protT5.

Env
=============
protT5 environment https://github.com/HannesStark/protein-localization.
```
# Pytorch==1.8.0 CUDA Version: 11.4 
conda env create -f env.yml
pip install -r requirements.txt
conda activate PLM_DBPPred
```
Using PLM_Sol
=============

Used the bio-embedding to generate the .h5 file
```
bio_embeddings light_attention.yaml
```
Training
```
python train.py --config ./configs/SOL_biLSTM_TextCNN
```
Predict
```
python predict.py --config ./configs/inference-SOL-test-LSTM-CNN.yaml  
```

