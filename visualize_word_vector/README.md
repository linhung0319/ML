# Visualize Word Vector

## Get Started

> 以 python 2.7.15 執行
> 
> gensim 3.6.0 </br>
> jieba 0.39 </br>
> sklearn 0.19.2 </br>
> matplotlib 2.23 </br>
> adjustText

## Dataset

> 578810條[中文句子](https://drive.google.com/file/d/1E5lElPutaWqKYPhSYLmVfw6olHjKDgdK/view)

## Preprocessing

ex: line = 「你喜歡吃水果嗎？」

1. **去除掉中文標點符號**

~~~~
line = rm_ch_punctuation(line)
~~~~
『你喜歡吃水果嗎』

2. **以詞斷句** 

~~~~
line = jieba.cut(line, cut_all=False)
~~~~
[你, 喜歡, 吃, 水果, 嗎]

## Word2Vec

~~~~
model = Word2Vec(sent_matrix, size=embd_size, window=s_window)
~~~~

使用gensim的Word2Vec，CBOW模式訓練Model

emnd_size = 300, 表示將

## Results

~~~~
y1 = model.similarity(u'好', u'不錯')
~~~~
> [好] 和 [不錯] 的 相似度 0.318412333727

計算 [好] 和 [不錯] 的 cosine similarity

發現在這個dataset，這兩個詞彙的相關性竟然不高，這可能跟data量仍然不夠多有關

~~~~
y2 = model.most_similar(u'書', topn=3)
~~~~
> 和 [書] 最相關的詞
> 採購, 0.733585834503
> 器材, 0.707304060459
> 文件, 0.704823195934

書可以被採購，因此相關性高

文件和書都是用來紀載資訊，因此相關性高

~~~~
y3 = model.most_similar([u'爸', u'女'], [u'媽'], topn=3)
~~~~
> [爸] - [媽] = [] - [女]
> 老牌子, 0.557583272457
> 佛羅倫, 0.554064154625
> 一流, 0.550119996071

兩向量相間，預計空格處應出現『男』，但結果卻不是，可能跟data量不夠多有關

## Reference
> 此題目和Dataset來均來自 [NTUEE Hung-yi Lee Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
