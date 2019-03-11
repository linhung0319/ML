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

> 以下為[Dataset](https://drive.google.com/file/d/1E5lElPutaWqKYPhSYLmVfw6olHjKDgdK/view)片段，總共578810條中文句子

~~~~
沒有我得不到的
現在你不誇獎我
你要把握這個機會
別在這裡丟人現眼
當然要知道他的想法
鐵的紀律的崇尚者對吧
不要牽扯我身邊的人
眼睜睜看她被你欺負
文進你趕快打電話回去
~~~~

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

可以理解為使用周圍的詞，來預測中間的詞為何。反過來說，語義相近的詞，其在句子中的周圍詞，也應該會類似

![](https://github.com/linhung0319/ML/blob/master/visualize_word_vector/cbow_skip-gram.jpeg)

emnd_size = 300, 將原本以one-hot encoding表示，每維代表一個詞彙的向量，投影到維度為300的座標系，其中相近語義的詞向量，會投影到相近的位置

s_window = 4，以周圍的4個詞來預測中間詞

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

![](https://github.com/linhung0319/ML/blob/master/visualize_word_vector/tsne_embd300.png)

使用TSNE將300維的詞向量，投影到2維，其中只選擇出現頻率在3000 - 6000的詞彙

經過觀察可以發現：

1.媽媽，媽，爸爸，爸，的位置相近

2.覺得，看到，聽，的位置相近


## Reference
> 題目和Dataset來均來自 [NTUEE Hung-yi Lee Course Website](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
>
> [輕鬆理解CBOW模型](https://blog.csdn.net/u010665216/article/details/78724856)
>
> [解決 matplotlib 中文圖片問題](https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b?fbclid=IwAR2KWGr7sVGLJR8xG3ZbGWwMBwEVm2rhTjQDWKg_RtPjHoCD_TowlMIuYzc)


