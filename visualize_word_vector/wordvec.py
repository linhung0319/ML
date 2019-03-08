#!/usr/bin/env python2
#-*- coding: UTF-8 -*-

import jieba
from gensim.models.word2vec import Word2Vec
import os
import numpy as np
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text


# For Chinese text in matplotlib
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def read_doc(doc_name):
    docs = []
    with open(doc_name, 'r') as f:
        for line in f.readlines():
            # Remove chinese punctuation
            line = rm_ch_punctuation(line)
            docs.append(line)
    return docs

def sentence_cut(docs):
    jieba.set_dictionary('dict.txt.big')
    sent_matrix = []
    for doc in docs:
        # Cut a sentence into words
        doc_iterator = jieba.cut(doc, cut_all=False)
        doc_list = [x for x in doc_iterator]
        sent_matrix.append(doc_list)
    return sent_matrix

def find_freq_words(model, freq_max, freq_min):
    word_list = []
    for word, vocab in model.wv.vocab.items():
        if freq_max and freq_min:
            if (vocab.count <= freq_max and
                vocab.count >= freq_min):
                word_list.append(word)
        elif freq_max:
            if (vocab.count <= freq_max):
                word_list.append(word)
        elif freq_min:
            if (vocab.count >= freq_min):
                word_list.append(word)
        else:
            word_list.append(word)
    return word_list

def rm_ch_punctuation(line):
    punc = """，！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    line = re.sub('[{}]'.format(punc).decode('utf-8'), '', line.decode('utf-8'))
    return line

def tsne(data):
  model = TSNE(n_components=2, random_state=0)
  return model.fit_transform(data)

def plot_fig(vectors, vocabs, filename='default.png'):
  # print(vectors.shape)
  # print(len(vocabs))
  xs, ys = vectors[:,0], vectors[:, 1]
  xs *= 10000
  ys *= 10000
  texts = []
  fig, ax = plt.subplots(figsize=(16, 16))
  for x, y, vocab in zip(xs, ys, vocabs):
    ax.plot(x, y, '.')
    texts.append(plt.text(x, y, vocab, fontsize=16))
  adjust_text(texts, arrowprops=dict(arrowstyle='-'))
  plt.xticks(np.array([]))
  plt.yticks(np.array([]))
  fig.savefig(filename)
  plt.close()

def main():
    embd_size = 300
    s_window = 5
    freq_max = 6000
    freq_min = 3000
    model_name = 'md_embd{}.model'.format(embd_size)
    if not os.path.exists(model_name):
        docs = read_doc("all_sents.txt")
        sent_matrix = sentence_cut(docs)

        model = Word2Vec(sent_matrix, size=embd_size, window=s_window)
        model.save(model_name)
    else:
        model = Word2Vec.load(model_name)

    word_list = find_freq_words(model,
                                freq_max=6000,
                                freq_min=3000)
    word_matrix = np.array([model.wv[x] for x in word_list])
    tsne_plane = tsne(word_matrix)
    plot_fig(tsne_plane, word_list, 'tsne_embd{}.png'.format(embd_size))

    with open('results_embd{}.txt'.format(embd_size), 'w') as fp:
        fp.write(u"model 內總共有 {} 種字\n".format(len(model.wv.vocab)))
        fp.write(u"介於出現頻率 {} - {} 共有 {} 種字\n\n".format(freq_min, freq_max, len(word_list)))

        y1 = model.similarity(u'好', u'不錯')
        fp.write(u"[好] 和 [不錯] 的 相似度 {}\n\n".format(y1))

        y2 = model.most_similar(u'書', topn=3)
        fp.write( u"和 [書] 最相關的詞：\n" )
        for i in y2:
            fp.write( "{}, {}\n".format(i[0], i[1]) )
        fp.write("\n")

        y3 = model.most_similar([u'爸', u'女'], [u'媽'], topn=3)
        fp.write( u"[爸] - [媽] = [] - [女]\n" )
        for i in y3:
            fp.write( "{}, {}\n".format(i[0], i[1]) )
        fp.write("\n")

if __name__ == '__main__':
    main()
