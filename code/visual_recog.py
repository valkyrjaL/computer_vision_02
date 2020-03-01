import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import matplotlib.pyplot as plt
import skimage.color

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    SPM_layer_num = 3
    K = dictionary.shape[0]
    M = K * int((4**(SPM_layer_num)-1)/3)
    features = np.empty((0, M))
    for idx, file_name in enumerate(train_data['files']):
        file_path = "../data/" + file_name
        print("#", idx, " : ", file_path)
        features = np.append(features, get_image_feature(file_path,dictionary,SPM_layer_num,K).reshape((1, M)), axis = 0)
        # if idx == 99:
        #     break
    labels = train_data["labels"]
    print(features.shape)
    print(labels.shape)
    print(dictionary.shape)
    np.savez("trained_system", features = features, labels = labels, dictionary = dictionary, SPM_layer_num = SPM_layer_num)

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    histograms = trained_system["features"]
    dictionary = trained_system["dictionary"]
    K = dictionary.shape[0]
    SPM_layer_num = trained_system["SPM_layer_num"]
    predicted_labels = np.array([], dtype = "int64")
    conf_mat = np.zeros((8,8))
    for idx, file_name in enumerate(test_data['files']):
        file_path = "../data/" + file_name
        print("#", idx, " : ", file_name)
        word_hist = get_image_feature(file_path,dictionary,SPM_layer_num,K)#.reshape((1, M))
        sim = distance_to_set(word_hist,histograms)
        p_label = trained_system["labels"][np.argmax(sim)]
        t_label = test_data["labels"][idx]
        # print("P:", p_label, "T:", t_label)
        conf_mat[t_label][p_label] += 1
        predicted_labels = np.append(predicted_labels, p_label)
    np.save("predicted_labels.npy", predicted_labels)
    np.save("confusion_matrix.npy", conf_mat)
    accuracy = np.trace(conf_mat) / sum(sum(conf_mat))
    print("Confusion Matrix: ")
    print(conf_mat)
    print("Accuracy: ", accuracy)
    return conf_mat, accuracy



def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)/3))
    '''

    # ----- TODO -----
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image,dictionary)
    return get_feature_from_wordmap_SPM(wordmap,layer_num,K)



def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    N = histograms.shape[0]
    aug_word = np.tile(word_hist, (N, 1))
    
    # sum(aug_word[aug_word < histograms]) + sum(histograms[histograms < aug_word])
    A = np.sum(np.where(aug_word < histograms, aug_word, 0.0), axis = 1)
    B = np.sum(np.where(aug_word > histograms, histograms, 0.0), axis = 1)
    sim = A + B
    return sim



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    hist, bin_edges = np.histogram(wordmap.flatten(), bins = range(dict_size + 1), density = True)
    # print("sum: ", hist.sum(), " / shape: ", hist.shape)
    # plt.hist(wordmap.flatten(), bins = dict_size, density = True)
    # plt.show()
    return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3) (as given in the write-up)
    '''
    
    # ----- TODO -----
    hist_num = int((4**(layer_num)-1)/3)
    hist_all = np.zeros((hist_num * dict_size))
    image_shape = wordmap.shape
    cell_num_sqrt = []
    weight = []
    for i in range(layer_num):
        cell_num_sqrt.append(2 ** i)
        if i < 2:
            weight.append(np.power(2, float(-(layer_num -1))))
        else:
            weight.append(np.power(2, float(i - (layer_num -1) -1)))
    cell_cum_idx = [0] + cell_num_sqrt[:-1]
    for i in range(len(cell_cum_idx) - 1):
        cell_cum_idx[i+1] = cell_cum_idx[i+1]**2 * dict_size + cell_cum_idx[i]
    
    '''start from the finest layer'''
    level = layer_num - 1
    cell = cell_num_sqrt[level]
    idx = cell_cum_idx[level]
    block_shape = [int(image_shape[0] / cell), int(image_shape[1] / cell)]
    for i in range(cell):
        for j in range(cell):
            block = wordmap[i * block_shape[0] : (i+1) * block_shape[0], j * block_shape[1] : (j+1) * block_shape[1]]
            hist = get_feature_from_wordmap(block,dict_size) / (cell ** 2)
            hist_all[idx : idx + dict_size] = hist
            idx += dict_size

    '''construct all other layers'''
    while level > 0:
        level -= 1
        cell = cell_num_sqrt[level]
        idx = cell_cum_idx[level]
        for i in range(cell):
            for j in range(cell):
                A = cell_cum_idx[level+1] + 2 * dict_size * (i*cell_num_sqrt[level+1] + j)
                B = A + dict_size * cell_num_sqrt[level+1]
                hist_all[idx : idx + dict_size] = \
                hist_all[A: A + dict_size] + hist_all[A + dict_size : A + 2*dict_size] + \
                hist_all[B : B + dict_size] + hist_all[B + dict_size : B + 2*dict_size]
                idx += dict_size

    '''adjust weightings'''
    for i in range(layer_num - 1):
        hist_all[cell_cum_idx[i] : cell_cum_idx[i+1]] *= weight[i]
    hist_all[cell_cum_idx[-1]:] *= weight[-1]
    # print(cell_num_sqrt, cell_cum_idx, weight)
    # print(image_shape, block_shape)
    # print("hist sum: ", hist_all.sum())
    return hist_all




    

