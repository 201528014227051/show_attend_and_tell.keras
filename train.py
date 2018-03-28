from evaluator import Evaluator
from generator import Generator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from models import NIC
from data_manager import DataManager
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
num_epochs = 1000
batch_size = 2#100#256
root_path = '../datasets/rsicd/'
captions_filename = root_path + 'origin_data.txt'
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=30,
                            word_frequency_threshold=2,
                            extract_image_features=False,
                            cnn_extractor='vgg16',
                            image_directory='/home/user2/qubo_captions/data/RSICD/imgs/',
                            split_data=False,
                            dump_path=root_path)

data_manager.preprocess()
print(data_manager.captions[0])
print(data_manager.word_frequencies[0:20])

preprocessed_data_path = root_path 
generator = Generator(data_path=preprocessed_data_path,
                      batch_size=batch_size)

num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of validation samples:', num_validation_samples)

model = NIC(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            rnn='lstm',
            num_image_features=generator.IMG_FEATS,
            hidden_size=128,
            embedding_size=128,
            batch_size=batch_size)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print('Number of parameters:', model.count_params())

training_history_filename = preprocessed_data_path + 'training_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('../trained_models/rsicd_attend/' +
               'rsicd_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [csv_logger, model_checkpoint, reduce_learning_rate]

model.fit_generator(generator=generator.flow(mode='train'),
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=int(num_validation_samples / batch_size))

#evaluator = Evaluator(model, data_path=preprocessed_data_path,
#                      images_path='/home/user2/qubo_captions/data/RSICD/imgs/')

#evaluator.write_captions()
#evaluator.display_caption()
