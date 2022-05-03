# LEX-GAN

Environment Setup

```shell
pip install tensorflow==2.3.1
pip install matplotlib pandas imbalanced-learn scikit-learn
conda install cudatoolkit=10.1 cudnn=7.6
```

Download GloVe embeddings and extract `.txt` files to `data/pre_embeddings` folder

Training
```shell
bash scripts/pheme_train.sh NOT_HOLD-ONE 0
```

## Please our work here if you used our data/code/materials:  
Cheng, Mingxi, Yizhi Li, Shahin Nazarian, and Paul Bogdan. "From rumor to genetic mutation detection with explanations: a GAN approach." Scientific Reports 11, no. 1 (2021): 1-14.
