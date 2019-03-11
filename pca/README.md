# Principal Components Analysis

## Get Started

> 以 python 2.7.15 執行
> 
> numpy 1.15.1 </br>
> skimage 0.14.0 </br>

## Dataset

> [CMU AMP Lab facial expression database](http://chenlab.ece.cornell.edu/projects/FaceAuthentication/download.html)

## Preprocessing

讀取所有圖片成為一個二維矩陣 (samples, features) = (975, 64 x 64)

總共975張圖片，每張圖片擁有64 x 64 Pixels

## PCA

X.shape = (samples, features) = (975, 4096) </br> 
X = X - mean(X, axis=0) </br></br>

欲求cov(X)的 eigenvalue 和 eigenvector </br>
cov(X) = X<sup>T</sup>X = V S V<sup>T</sup> </br>
V = eigenvector </br>
S = eigenvelue </br></br>
對 X 使用Singular Value Decomposition </br>
X = U s V<sup>T</sup>
X<sup>T</sup>X = V s U<sup>T</sup> U s V<sup>T</sup> = V s<sup>2</sup> V<sup>T</sup> </br></br>
因此對X使用Singular Value Decomposion，可以得到 eigenvector 和 eigenvalue 

~~~~
x = x - mu
u, sigma, eigen_faces = np.linalg.svd(x, full_matrices=False)
~~~~

## Results

![](https://github.com/linhung0319/ML/blob/master/pca/mean_face.png)

所有臉的平均, mean(X, axis=0)

![](https://github.com/linhung0319/ML/blob/master/pca/eigen_faces.png)

前9大 eigenvalue 對應的eigenvector

~~~
Weighting of the 1 eigenvalue = 0.0513798413302
Weighting of the 2 eigenvalue = 0.039747754919
Weighting of the 3 eigenvalue = 0.0327542422395
Weighting of the 4 eigenvalue = 0.0291679047803
Weighting of the 5 eigenvalue = 0.0269437848274
Weighting of the 6 eigenvalue = 0.0215428903316
Weighting of the 7 eigenvalue = 0.0198333507036
Weighting of the 8 eigenvalue = 0.0177730117559
Weighting of the 9 eigenvalue = 0.0166422270234
~~~

![](https://github.com/linhung0319/ML/blob/master/pca/original_faces.png)

選取16張圖片

![](https://github.com/linhung0319/ML/blob/master/pca/reconstructed_faces.png)

選取前9大 eigenvalue 對應的 eigenvector 重建圖片

可以發現雖然圖片變模糊了，但是仍然能夠看出原來圖片的樣子，表示精簡後的維度

## Reference

> 題目和Dataset來均來自 [NTUEE Hung-yi Lee Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
>
> [機器/統計學習:主成分分析(Principal Component Analysis, PCA)](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%BF%92-%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90-principle-component-analysis-pca-58229cd26e71)
