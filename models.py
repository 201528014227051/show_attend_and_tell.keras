from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense, Lambda, Permute, Convolution2D
from keras.layers import BatchNormalization, Embedding, Activation, Reshape, Multiply, RepeatVector
from keras.layers.merge import Add, Concatenate, Average
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2
from attendlayer import AttendLayer, AVE ,DenseLast
from keras import backend as K

def NIC(max_token_length, vocabulary_size, rnn='lstm' ,num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8, batch_size= 20):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), name='text')#batch_shape=batch_size,
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding
    image_input = Input(shape=(max_token_length, 14, 14, 512), #batch_shape=batch_size,
                                                        name='image')
    #image_input2 = Reshape((max_token_length, 196, 512), name='reshape1')(image_input)
    
    image_input2 = TimeDistributed(Convolution2D(1, (3,3), padding='same', name = 'conv1'))(image_input)
    image_input3 = Reshape((max_token_length, 196), name='reshape1')(image_input2)
  #  denselastlayer
    image_input3_a = TimeDistributed(Dense(units=196,
                                      kernel_regularizer=l2(regularizer),
                                      activation = 'softmax',
                                      name='image_weight'))(image_input3)
    image_weight = Activation('softmax', name='activation1')(image_input3_a)
    image_weight_r = TimeDistributed(RepeatVector(512, name='repeatv'))(image_weight)
    image_weight2 = Reshape((max_token_length, 14, 14, 512), name='reshape2')(image_weight_r)
    image_embedding_p = Multiply(name='multi')([image_input, image_weight2])
    image_embedding_p2 = TimeDistributed(Convolution2D(1, (3,3), padding='same', name = 'conv2'))(image_embedding_p)
    image_embedding_p3 = Reshape((max_token_length, 196), name='reshape3')(image_embedding_p2)
    image_embedding = TimeDistributed(Dense(units=embedding_size,
                                      kernel_regularizer=l2(regularizer),
                                      activation = 'softmax',
                                      name='image_embedding'))(image_embedding_p3)
    image_dropout = Dropout(.5, name='image_dropout')(image_embedding)
    #perted = Permute((1, 3, 2), name = 'permute')(image_dropout)
    # language model
    recurrent_inputs = [text_dropout, image_dropout]
    merged_input = Add()(recurrent_inputs)
    if rnn == 'lstm':
        
        lstm_out = LSTM(units=hidden_size,#[:, i, :]
                        recurrent_regularizer=l2(regularizer),
                        kernel_regularizer=l2(regularizer),
                        bias_regularizer=l2(regularizer),
                        return_sequences=True,
                        name='recurrent_network')(merged_input)

    else:
        raise Exception('Invalid rnn name')

    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(lstm_out)

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = NIC(16, 1024)
    plot_model(model, './images/NIC2.png')
