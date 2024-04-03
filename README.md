# PLM_Sol
Cite the code: [![DOI](https://zenodo.org/badge/743842028.svg)](https://zenodo.org/doi/10.5281/zenodo.10675340)

What is PLM_Sol?
=============
A protein solubility prediction tool based on protT5 and biLSTM_textCNN.

Env
=============
protT5 environment https://github.com/HannesStark/protein-localization.
```
# Pytorch==2.0.1 CUDA Version: 11.4 
conda env create -f env.yml
conda activate PLM_Sol
cd ./PLM_Sol
pip install -r requirements.txt
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

