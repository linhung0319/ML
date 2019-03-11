#!/usr/bin/env python2

import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

#np.set_printoptions(threshold=np.nan)

def read_pic(pic_dir):
    pics = []
    pics_n = [x for x in os.listdir(pic_dir) if x.endswith('.bmp')]
    for pic_n in pics_n:
        pic = io.imread(os.path.join(pic_dir, pic_n))
        pics.append(pic.flatten())
    width = pic.shape[0]
    height = pic.shape[1]
    return np.array(pics), width, height

def pca(x, picked=9):
    mu = np.mean(x, axis=0)
    print "Shape of mu = {}".format(mu.shape)
    x = x - mu
    u, sigma, eigen_faces = np.linalg.svd(x, full_matrices=False)

    print ""
    print "Singular Value Decomposition:"
    print "Shape of u = {}".format(u.shape)
    print "Shape of sigma = {}".format(sigma.shape)
    print "Shape of vT = {}".format(eigen_faces.shape)

    picked_faces = eigen_faces[:picked]
    weights = np.dot(x, picked_faces.T)
    return mu, weights, picked_faces, sigma

def reconstruct(mu, weights, eigen_face):
    rec_faces = np.dot(weights, eigen_face)
    rec_faces += mu
    return rec_faces

def save_img(data, width, height,
             filename='default.png', subplot=False, size=0):
    if not subplot:
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(data.reshape(width, height), cmap='gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()
    else:
        fig_width = np.ceil(np.sqrt(size))
        fig = plt.figure(figsize=(16, 16))
        for i in range(size):
            ax = fig.add_subplot(fig_width, fig_width, i+1)
            ax.imshow(data[i].reshape(width, height), cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.savefig(filename)
        plt.close()

def main():
    pic_dir = 'face'
    pics_matrix, width, height = read_pic(pic_dir)
    print "picture\'s width = {}".format(width)
    print "picture\'s height = {}".format(height)
    print "pics_matrix 0-axis dimension = {}".format(pics_matrix.shape[0])
    print "pics_matrix 1-axis dimension = {}".format(pics_matrix.shape[1])

    mu, weights, eigen_faces, sigma = pca(pics_matrix)
    rec_faces = reconstruct(mu, weights, eigen_faces)

    pick_eigenvalues = 4
    print ""
    for i in range(pick_eigenvalues):
        print "Weighting of the {} eigenvalue = {}".format(i + 1, sigma[i] / np.sum(sigma))

    # Save mean face
    save_img(mu, width, height, filename='mean_face.png')
    # Save eigen faces
    save_img(eigen_faces, width, height,
             filename='eigen_faces.png', subplot=True, size=eigen_faces.shape[0])
    # Save original faces and reconstucted faces
    faces_num = 16
    save_img(pics_matrix[:faces_num], width, height,
             filename='original_faces.png', subplot=True, size=faces_num)
    save_img(rec_faces[:faces_num], width, height,
             filename='reconstructed_faces.png', subplot=True, size=faces_num)


if __name__ == '__main__':
    main()
