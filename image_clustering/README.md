# Image Clustering

使用Unsupervised Learning的方式，讓機器自己學會分辨兩組沒有被Labeled的Dataset

在Training Data上分別以AutoEncoder和PCA降維，再使用KMeans執行Clustering

比較在Training Data上學習到AutoEncoder + KMeans和PCA + KMeans，與直接在Testing Data上使用TSNE + KMeans的表現

## Getting Started

> 以 python 2.7.15 執行
> 
> numpy 1.15.1 </br>
> matplotlib 2.23 </br>
> sklearn 0.19.2 </br>
> keras 2.24 </br>
> tensorflow 1.12.0-rc0

## PCA + KMeans

~~~~
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
~~~~

~~~~
pca_model = PCA(n_components=embd_dim).fit(data)

x_pca = pca_model.transform(data)
pca_km_model = KMeans(n_clusters=2).fit(x_pca)
~~~~

使用PCA降至60維，再使用Kmeans將Data分類為兩類

![](https://github.com/linhung0319/ML/blob/master/image_clustering/pca_2dim.png)

取PCA的前兩維作圖，可以發現『數字』圖片和『衣物』圖片有被分開的趨勢，但仍然有部份互相重疊

![](https://github.com/linhung0319/ML/blob/master/image_clustering/pca_img.png)

隨機取36張圖片觀察預測結果，發現『數字』圖片被很好的分類，然而某些『衣物』圖片被誤判

| Confusion Matrix | Predicted Number | Predicted Clothes |
|:-----------------|:-----------------|:------------------|
| True Number      | 4986 (TP)        | 14   (FP)         |
| True Clothes     | 2449 (FN)        | 2551 (TN)         |

預測準確率75.37%，『衣物』圖片有50%的機率被判斷錯誤

## AutoEncoder + KMeans

~~~~
from keras.layers import Input, Dense
from keras.models import Model
~~~~

~~~~
class AutoEncoder():
    def __init__(self, embd_dim, img_dim=784):
        self.img_dim = img_dim
        self.embd_dim = embd_dim
        self.forward()

    def forward(self):
        self.__input_img = Input(shape=(self.img_dim,))
        self.__encoded = Dense(self.embd_dim, activation='relu', name='encoder')(self.__input_img)
        self.__decoded = Dense(self.img_dim, activation='sigmoid', name='decoder')(self.__encoded)
        self.AE_model = Model(self.__input_img, self.__decoded)
        self.AE_model.compile(optimizer='adadelta', loss='binary_crossentropy')

    @property
    def encoder(self):
        return Model(self.__input_img, self.__encoded)

    @property
    def decoder(self):
        input_decoder = Input(shape=(self.embd_dim,))
        output = self.AE_model.get_layer('decoder')(input_decoder)
        return Model(input_decoder, output)

    def load_weights(self, fname):
        self.AE_model.load_weights(fname, by_name=True)
~~~~

![](https://github.com/linhung0319/ML/blob/master/image_clustering/ae_architecture.png)

AutoEncoder將圖片降至60維，再以60維的資訊重建圖片，希望重建圖片能夠與原圖片越接近

使用Binary Cross Entropy作為Loss Function，將每個Pixel視為一個獨立的分佈來判斷重建圖片與原圖片是否相近

事實上使用Binary Cross Entropy或Mean Squared Error作為Loss Function都不能很正確的作為圖片是否相近的依據，因為他們沒有考慮到Pixel與Pixel間的關係。GAN使用的Discrimanator就是想要盡可能的解決如何判斷兩個物件是真是假的問題

![](https://github.com/linhung0319/ML/blob/master/image_clustering/ae_2dim.png)

取AutoEncoder的Embedding Vector前兩維作圖

![](https://github.com/linhung0319/ML/blob/master/image_clustering/ae_img.png)

AutoEncoder也有把『衣物』圖片誤判為『數字』圖片的問題，或許可以藉由Data Augmentaion增加Training Data數量，或是訓練更多epoch來增加AutoEncoder萃取Feature的能力

| Confusion Matrix | Predicted Number | Predicted Clothes |
|:-----------------|:-----------------|:------------------|
| True Number      | 4702 (TP)        | 298  (FP)         |
| True Clothes     | 2349 (FN)        | 2651 (TN)         |

預測準確率73.53%，稍低於PCA的準確率，然而其預測『衣物』

## Reference

> 題目和Dataset來均來自 [NTUEE Hung-yi Lee Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
