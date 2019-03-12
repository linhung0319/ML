#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Adapt GPU allocate
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def load_img(dataset):
    x = np.load(dataset)
    return x

def plot_scatter(x_true, y_true,
                 x_pred, y_pred, filename='default.png'):
    fig = plt.figure(figsize=(16, 8))
    c_true = label_to_color(y_true)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x_true[:, 0], x_true[:, 1], c=c_true, s=0.5)
    plt.title("true")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    c_pred = label_to_color(y_pred)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x_pred[:, 0], x_pred[:, 1], c=c_pred, s=0.5)
    plt.title("predict")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(filename)
    plt.close()

def plot_imgs(data, width, height, label=[], filename='default.png', size=1):
    fig_width = np.ceil(np.sqrt(size))
    fig = plt.figure(figsize=(16, 16))
    for i in range(size):
        ax = fig.add_subplot(fig_width, fig_width, i+1)
        ax.imshow(data[i].reshape(width, height), cmap='gray')
        if label.any():
            plt.title('class {}'.format(label[i]), fontsize=20)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.savefig(filename)
    plt.close()

def plot_table(matrix, filename='default.png'):
    rows = ['true 0', 'true 1']
    cols = ['pred 0', 'pred 1']
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=matrix, colLabels=cols, rowLabels=rows, loc='center')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

def label_to_color(label):
    c = []
    for lab in label:
        if lab == 0:
            c.append('r')
        elif lab == 1:
            c.append('b')
        else:
            c.append('g')
    return c

def train_pca(data, embd_dim=60, pca_md='pca_model.p', pca_km_md='pca_km.p'):
    if not os.path.exists(pca_md):
        pca_model = PCA(n_components=embd_dim).fit(data)
        pickle.dump(pca_model, open(pca_md, 'wb'))

    if not os.path.exists(pca_km_md):
        pca_model = pickle.load(open(pca_md, 'rb'))
        x_pca = pca_model.transform(data)
        pca_km_model = KMeans(n_clusters=2).fit(x_pca)
        pickle.dump(pca_km_model, open(pca_km_md, 'wb'))

def train_AutoEncoder(data, embd_dim, ae_md='ae_model.weight', ae_km_md='ae_km.p'):
    pic_matrix = normalize_img(data)
    split_ratio = 0.1
    train_num = int((1 - split_ratio) * pic_matrix.shape[0])
    pic_train = pic_matrix[:train_num]
    pic_test = pic_matrix[train_num:]

    ae = AutoEncoder(embd_dim)
    ae.AE_model.summary()
    if not os.path.exists(ae_md):
        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss'))
        callbacks.append(ModelCheckpoint(filepath=ae_md,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=True ))

        ae.AE_model.fit(pic_train, pic_train,
                        epochs=50,
                        batch_size=128,
                        shuffle=True,
                        validation_data=[pic_test, pic_test],
                        callbacks=callbacks)

    if not os.path.exists(ae_km_md):
        ae.load_weights(ae_md)
        x_ae = ae.encoder.predict(pic_matrix)
        ae_km_model = KMeans(n_clusters=2).fit(x_ae)
        pickle.dump(ae_km_model, open(ae_km_md, 'wb'))

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

def normalize_img(data):
    data = data.astype('float32') / 255
    return data

def test_pca(x, y, pca_md, pca_km_md):
    pic_width = 28
    print "Testing PCA..."
    pca_model = pickle.load(open(pca_md, 'rb'))
    x_pca = pca_model.transform(x)
    pca_km_model = pickle.load(open(pca_km_md, 'rb'))
    y_pca = pca_km_model.predict(x_pca)
    pca_cm = confusion_matrix(y, y_pca)
    plot_table(pca_cm, 'pca_cm.png')
    plot_scatter(x_pca[:,:2], y,
                 x_pca[:,:2], y_pca, filename='pca_2dim.png')
    plot_imgs(x, pic_width, pic_width,
              label=y_pca, filename='pca_img.png', size=36)

def test_AutoEncoder(x, y, embd_dim, ae_md, ae_km_md):
    pic_width = 28
    print "Testing AutoEncoder..."
    x = normalize_img(x)
    ae = AutoEncoder(embd_dim)
    ae.load_weights(ae_md)
    x_ae = ae.encoder.predict(x)
    ae_km_model = pickle.load(open(ae_km_md, 'rb'))
    y_ae = ae_km_model.predict(x_ae)
    ae_cm = confusion_matrix(y, y_ae)
    plot_table(ae_cm, 'ae_cm.png')
    plot_scatter(x_ae[:,:2], y,
                 x_ae[:,:2], y_ae, filename='ae_2dim.png')
    plot_imgs(x, pic_width, pic_width,
              label=y_ae, filename='ae_img.png', size=36)

def test_tsne(x, y):
    pic_width = 28
    print "processing TSNE..."
    x_tsne = TSNE(n_components=2, random_state=0).fit_transform(x)
    y_tsne = KMeans(n_clusters=2).fit_predict(x_tsne)
    tsne_cm = confusion_matrix(y, y_tsne)
    plot_table(tsne_cm, 'tsne_cm.png')
    plot_scatter(x_tsne[:,:2], y,
                 x_tsne[:,:2], y_tsne, filename='tsne_2dim.png')
    plot_imgs(x, pic_width, pic_width,
              label=y_tsne, filename='tsne_img.png', size=36)

if __name__ == '__main__':
    # Embedding dimension
    embd_dim = 60

    # Prepare Training Data
    pic_matrix = load_img('image.npy')

    # Train PCA
    pca_md = 'pca_model.p'
    pca_km_md = 'pca_km.p'
    print "Training PCA..."
    train_pca(pic_matrix, embd_dim, pca_md, pca_km_md)

    # Train AutoEncoder
    ae_md = 'ae_model.weight'
    ae_km_md = 'ae_km.p'
    print "Training AutoEncoder..."
    train_AutoEncoder(pic_matrix, embd_dim, ae_md, ae_km_md)

    # Prepare Testing Data
    pic_width = 28
    data_num = 10000
    x = load_img('visualization.npy')
    y = np.array([i / 5000 for i in range(data_num)])
    data_idx = np.array([i for i in range(data_num)])
    np.random.shuffle(data_idx)
    x = x[data_idx]
    y = y[data_idx]
    plot_imgs(x, pic_width, pic_width,
              label=y, filename='ori_img.png', size=36)

    # Test PCA
    test_pca(x, y, pca_md, pca_km_md)

    # Test AutoEncoder
    test_AutoEncoder(x, y, embd_dim, ae_md, ae_km_md)

    # Test TSNE
    test_tsne(x, y)
