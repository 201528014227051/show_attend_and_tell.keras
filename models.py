from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense, Lambda, Permute
from keras.layers import BatchNormalization, Embedding, Activation, Reshape
from keras.layers.merge import Add, Concatenate, Average
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2
from attendlayer import AttendLayer, AVE ,DenseLast
from keras import backend as K

def NIC(max_token_length, vocabulary_size, rnn='lstm' ,num_image_features=2048,
        hidden_size=512, embedding_size=512, regularizer=1e-8, batch_size= 20):

    # word embedding
    text_input = Input(shape=(max_token_length, vocabulary_size), batch_shape=batch_size, name='text')
    text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
    text_to_embedding = TimeDistributed(Dense(units=embedding_size,
                                        kernel_regularizer=l2(regularizer),
                                        name='text_embedding'))(text_mask)

    text_dropout = Dropout(.5, name='text_dropout')(text_to_embedding)

    # image embedding
    image_input = Input(shape=(max_token_length, 14, 14, 512), batch_shape=batch_size,
                                                        name='image')
    image_input2 = Reshape((max_token_length, 196, 512), name='reshape1')(image_input)
  #  denselastlayer
    image_embedding = TimeDistributed(Dense(units=embedding_size,
                                      kernel_regularizer=l2(regularizer),
                                      name='image_embedding'))(image_input2)
    image_dropout = Dropout(.5, name='image_dropout')(image_embedding)

    # language model
    #recurrent_inputs = [text_dropout, text_dropout]
   # merged_input = Add()(recurrent_inputs)
    if rnn == 'lstm':
        for i in range(max_token_length):
            if i == 0:
              #  first_input = AVE(units=1,
               ##                  name='initial_zero')(image_dropout[:, i, :, :])
                order_input = [image_dropout[:, i, j, :] for j in range(196)]
                first_input = Average()(order_input)
                #first_input = Dense(units=100,
                 #                   kernel_regularizer=l2(regularizer),
                  #                  name='initial_zero_embed')(first_input)
                recurrent_inputs = [text_dropout[:, i, :], first_input]
                merged_input_temp = Add()(recurrent_inputs)
                merged_input = Reshape((1, embedding_size), name='reshape2')(merged_input_temp)
            else:
                attendlayer = Lambda(lambda x: K.dot(x[0], x[1]))
                per_out = Permute((2, 1))(image_dropout[:, i, :, :])
                dim_change = Dense(units=196,
                                   kernel_regularizer=l2(regularizer),
                                   name='change_dim')(lstm_out)
                per_out2 = Permute((2, 1))(dim_change)
                for m in range(batch_size):#image_input.shape[0]
                    attendout = attendlayer([per_out[m, :, :], per_out2[m, :, :]])
                    if m == 0:
                        attendout2 = Concatenate(axis=1)([attendout, attendout])
                    else:
                        attendout2 = Concatenate(axis=1)([attendout2, attendout])
               # attendout = AttendLayer(units=1,
                ##                       name='attend_layer' + str(i))(lstm_out)
               # attendout2 = Reshape((-1, 128))(attendout)
                #attendout3 = Permute((1, 0))(attendout2[:, 1:])
                translayer = Lambda(lambda x: K.transpose(x))
                attendout3 = translayer(attendout2[:, 1:])
                recurrent_inputs = [text_dropout[:, i, :], attendout3]
                merged_input_temp = Add()(recurrent_inputs)
                merged_input = Reshape((1, embedding_size), name='reshape3')(merged_input_temp)
            lstm_out = LSTM(units=hidden_size,#[:, i, :]
                            recurrent_regularizer=l2(regularizer),
                            kernel_regularizer=l2(regularizer),
                            bias_regularizer=l2(regularizer),
                            return_sequences=True,
                            name='recurrent_network' + str(i))(merged_input)
            if i == 0:
                lstm_out_final = Concatenate(axis=1)([lstm_out, lstm_out])
            else:
                lstm_out_final = Concatenate(axis=1)([lstm_out_final, lstm_out])
    else:
        raise Exception('Invalid rnn name')
    output_be = Reshape((max_token_length+1, embedding_size))(lstm_out_final)
    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(output_be[:, 1:, :])

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = NIC(16, 1024)
    plot_model(model, '../images/NIC.png')
