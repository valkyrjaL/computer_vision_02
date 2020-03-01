import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # ----- TODO -----
    # start = time.time()
    filter_num = 4
    scales = [1, 2,4, 8, 8*2**(1/2)]
    image_shape = image.shape
    if len(image_shape) == 2:   # grayscale image
        image = np.stack([image, image, image], axis = 2)
    filter_responses = np.empty([image_shape[0], image_shape[1], 3 * filter_num * len(scales)]) # only use RGB channels, if alpha exists, ignore it
    image = skimage.color.rgb2lab(image)
    
    for i, scale_val in enumerate(scales):
        for channel in range(3):
            filter_responses[:,:,i*12 + channel] = scipy.ndimage.gaussian_filter(image[:,:,channel], sigma = scale_val, order = 0)
        for channel in range(3):
            filter_responses[:,:,i*12 + 3 + channel] = scipy.ndimage.gaussian_laplace(image[:,:,channel], sigma = scale_val)
        for channel in range(3):
            filter_responses[:,:,i*12 + 6 + channel] = scipy.ndimage.gaussian_filter(image[:,:,channel], sigma = scale_val, order = [0, 1])
        for channel in range(3):
            filter_responses[:,:,i*12 + 9 + channel] = scipy.ndimage.gaussian_filter(image[:,:,channel], sigma = scale_val, order = [1, 0])
    # end = time.time()
    # print("time: ", end - start)
    return filter_responses


def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    image_shape = image.shape
    wordmap = np.empty((image_shape[0], image_shape[1]))
    filter_responses = extract_filter_responses(image)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            wordmap[i][j] = np.argmin(scipy.spatial.distance.cdist(filter_responses[i, j, :].reshape((1, 60)), dictionary, "euclidean"))
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    i,alpha,image_path = args
    # ----- TODO -----
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    image_shape = image.shape
    filter_responses = extract_filter_responses(image)
    idx_0 = np.random.permutation(range(image_shape[0]))
    idx_1 = np.random.permutation(range(image_shape[1]))
    sampled_response = filter_responses[idx_0[:alpha], idx_1[:alpha]]
    temp_file = "../temp/sampled_response_" + str(i)
    np.savez(temp_file, sampled_response = sampled_response)
    # print(filter_responses.shape, image_shape)
    # print(sampled_response.shape)
    

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    NOTE : Please save the dictionary as 'dictionary.npy' in the same dir as the code.
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----    
    temp_path = "../temp"
    try:
        os.mkdir(temp_path)
    except OSError:
        print ("Creation of the directory %s failed" % temp_path)

    alpha = 100
    K = 200
    print(train_data['files'][0], train_data['labels'][0])
    for idx, file_name in enumerate(train_data['files']):   # approx 20 min
        path_img = "../data/" + file_name
        print("#", idx, " : ", path_img)
        compute_dictionary_one_image([idx, alpha, path_img])

    filter_responses = np.empty((0, 60))
    for i in range(train_data['files'].shape[0]):
        temp_file = "../temp/sampled_response_" + str(i) + ".npz"
        filter_responses = np.append(filter_responses, np.load(temp_file)["sampled_response"], axis = 0)
    print(filter_responses.shape)

    kmeans = sklearn.cluster.KMeans(n_clusters = K).fit(filter_responses)   # approx 17 min
    dictionary = kmeans.cluster_centers_
    dictionary_file = "dictionary.npy"
    np.save(dictionary_file, dictionary)
