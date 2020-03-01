import numpy as np
import torchvision
import util
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io

import time

if __name__ == '__main__':
    start = time.time()  #ctl
    num_cores = util.get_num_CPU()
    # path_img = "../data/kitchen/sun_avuzlcqxzrzteyvc.jpg"
    path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)
    # visual_words.compute_dictionary(num_workers=num_cores)

    dictionary = np.load('dictionary.npy')
    wordmap = visual_words.get_visual_words(image,dictionary)
    util.save_wordmap(wordmap, "wordmap_n")
    # visual_recog.get_feature_from_wordmap(wordmap,dictionary.shape[0])  #ctl
    # visual_recog.get_feature_from_wordmap_SPM(wordmap,3,dictionary.shape[0])  #ctl
    visual_recog.build_recognition_system(num_workers=num_cores)  # approx 12 min for 100 files, about 2 hr for 1000 files

    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores) # approx 1 hr for 577 files
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())
    end = time.time()  #ctl
    print("Time: ", end - start)  #ctl

