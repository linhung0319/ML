# Image Clustering

使用Unsupervised Learning的方式，讓機器自己學會分辨兩組沒有被Labeled的Dataset

在Training Data上分別以AutoEncoder和PCA降維，再使用KMeans執行Clustering

比較在Training Data上學習到AutoEncoder + KMeans和PCA + KMeans，與直接在Testing Data上使用TSNE + KMeans

## Getting Started

> 以 python 2.7.15 執行
> 
> numpy 1.15.1 </br>
> matplotlib 2.23 </br>
> sklearn 0.19.2 </br>
> keras 2.24 </br>
> tensorflow 1.12.0-rc0
