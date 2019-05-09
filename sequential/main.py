simport numpy as np
import torchvision
import util
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io

if __name__ == '__main__':

    num_cores = util.get_num_CPU()

    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    #visual_words.compute_dictionary(num_workers=num_cores)
    
    #dictionary = np.load('dictionary.npy')
    #img = visual_words.get_visual_words(image,dictionary)
    #util.save_wordmap(wordmap, filename)
    #visual_recog.build_recognition_system(num_workers=num_cores)

    #conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

