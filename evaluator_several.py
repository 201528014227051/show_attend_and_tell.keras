import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Evaluator(object):

    def __init__(self, model,
            data_path='preprocessed_data/',
            images_path='iaprtc12/',
            log_filename='data_parameters.log',
            test_data_filename='test_data.txt',
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename='vgg16_image_name_to_features.h5'):
        self.model = model
        self.data_path = data_path
        self.images_path = images_path
        self.log_filename = log_filename
        data_logs = self._load_log_file()
        self.BOS = str(data_logs['BOS:'])
        self.EOS = str(data_logs['EOS:'])
        self.IMG_FEATS = int(data_logs['IMG_FEATS:'])
        self.MAX_TOKEN_LENGTH = int(data_logs['max_caption_length:']) + 2
        self.test_data = pd.read_table(data_path +
                                       test_data_filename, sep='*')
        self.word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.image_names_to_features = h5py.File(data_path +
                                        image_name_to_features_filename)

    def _load_log_file(self):
        data_logs = np.genfromtxt(self.data_path + 'data_parameters.log',
                                  delimiter=' ', dtype='str')
        data_logs = dict(zip(data_logs[:, 0], data_logs[:, 1]))
        return data_logs


    def display_caption(self, image_file=None, data_name=None):

        if data_name == 'ad_2016':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('ad_2016')]
        elif data_name == 'iaprtc12':
            test_data = self.test_data[self.test_data['image_names'].\
                                            str.contains('iaprtc12')]
        else:
            test_data = self.test_data

        if image_file == None:
            image_name = np.asarray(test_data.sample(1))[0][0]
        else:
            image_name = image_file
        features = self.image_names_to_features[image_name]['image_features'][:]
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)
        for word_arg in range(self.MAX_TOKEN_LENGTH):
            predictions = self.model.predict([text, image_features])
            word_id = np.argmax(predictions[0, word_arg, :])
            next_word_arg = word_arg + 1
            text[0, next_word_arg, word_id] = 1
            word = self.id_to_word[word_id]
            print(word)
            if word == self.EOS:
                break
            #images_path = '../dataset/images/'
        plt.imshow(plt.imread(self.images_path + image_name))
        plt.show()

    def sub_process(self, predictions = None, word_arg = None, word_id = None):
    # delete the max by set to zero
        predictions[0, word_arg, word_id] = 0
        return predictions
        
    def write_captions_2d(self, dump_filename=None, id_im = 1):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_captions_' + str(id_im) + '.txt'

        predicted_captions = open(dump_filename, 'w')

        image_names = self.test_data['image_names'].tolist()
        count = 1
        for image_name in image_names:
            #print(count)
            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features
            neural_caption = []
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                predictions = self.model.predict([text, image_features])
                # print(predictions[0, word_arg, :])
                # print('max:%f' % np.max(predictions[0, word_arg, :]))
                # print('min:%f' % np.min(predictions[0, word_arg, :]))
                if word_arg == 0:
                    for i in range(id_im):
                        word_id = np.argmax(predictions[0, word_arg, :])
                    #predictions[0, word_arg, word_id] = 0
                        predictions = self.sub_process(predictions=predictions, word_arg=word_arg, word_id=word_id)
                    word_id = np.argmax(predictions[0, word_arg, :])
                else:
                    word_id = np.argmax(predictions[0, word_arg, :])
                next_word_arg = word_arg + 1
                text[0, next_word_arg, word_id] = 1
                word = self.id_to_word[word_id]
                if word == '<E>':
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            count += 1
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        print('%d done.' % id_im)
        # target_captions = self.test_data['caption']
        # target_captions.to_csv(self.data_path + 'target_captions.txt',
                               # header=False, index=False)

    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_captions.txt'

        predicted_captions = open(dump_filename, 'w')

        image_names = self.test_data['image_names'].tolist()
        for image_name in image_names:

            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features
            neural_caption = []
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                predictions = self.model.predict([text, image_features])
                word_id = np.argmax(predictions[0, word_arg, :])
                next_word_arg = word_arg + 1
                text[0, next_word_arg, word_id] = 1
                word = self.id_to_word[word_id]
                if word == '<E>':
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['caption']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)

if __name__ == '__main__':
    from keras.models import load_model

    root_path = '../datasets/rsicd/'
    data_path = root_path 
    images_path = '/home/user2/qubo_captions/data/RSICD/imgs/'
    model_filename = '../trained_models/rsicd2/rsicd_weights.458-2.00.hdf5'
    model = load_model(model_filename)
    evaluator = Evaluator(model, data_path, images_path)
    #evaluator.write_captions()
    print('first done.')
    for i in range(1, 10):
        evaluator.write_captions_2d(id_im = i)
    #evaluator.display_caption()
