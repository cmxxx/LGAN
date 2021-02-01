# LGAN

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