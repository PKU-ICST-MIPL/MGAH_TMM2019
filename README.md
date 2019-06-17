# Introduction
This is the source code of our TMM 2019 paper "Multi-pathway Generative Adversarial Hashing for Unsupervised Cross-modal Retrieval", Please cite the following paper if you use our code.

Jian Zhang and Yuxin Peng, “Multi-pathway Generative Adversarial Hashing for Unsupervised Cross-modal Retrieval”, IEEE Transactions on Multimedia (TMM), DOI:10.1109/TMM.2019.2922128, 2019. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201915)

# Usage
For PKU-Xmeida dataset:

1. Generate KNN graph by the codes under KNN directory: /xmedia/python/knn_5M.py
2. Train the model by using the code under unsuper-pretrain-xm-5M: python train_argv.py hashdim gpuid

hashdim represents the length of hash codes, gpuid represts the index of gpu

# Tips: 
You can download the data in /media from [download Link](https://pan.baidu.com/s/1VDDfEQDeCQKqBHEU8Xav-Q) pw:0q8o

For 2-media datasets codes, please refer to our [AAAI paper](https://github.com/PKU-ICST-MIPL/UGACH_AAAI2018).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
