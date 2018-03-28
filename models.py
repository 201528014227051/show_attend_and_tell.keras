from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense, Lambda, Permute
from keras.layers import BatchNormalization, Embedding, Activation, Reshape, Multiply
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
                getlistlayer = Lambda(lambda x: [x[:, j, :] for j in range(196)])
                getim0layer = Lambda(lambda x: x[:, 0, :, :])
                temp_layer = getim0layer(image_dropout)
                #order_input = getlistlayer(temp_layer)
                #order_input = [image_dropout[:, i, j, :] for j in range(196)]
                avelayer = Lambda(lambda x:K.mean(x, axis=1))
                first_input = avelayer(temp_layer)
                #first_input = Average()(order_input)
                #first_input = Dense(units=100,
                 #                   kernel_regularizer=l2(regularizer),
                  #                  name='initial_zero_embed')(first_input)
                gettx0layer = Lambda(lambda x: x[:, 0, :])
                first_input2 = gettx0layer(text_dropout)
                recurrent_inputs = [first_input2, first_input]
                merged_input_temp = Add()(recurrent_inputs)
                merged_input = Reshape((1, embedding_size), name='reshape2')(merged_input_temp)
            else:
                getim1layer = Lambda(lambda x: x[:, 1, :, :])
                getim2layer = Lambda(lambda x: x[:, 2, :, :])
                getim3layer = Lambda(lambda x: x[:, 3, :, :])
                getim4layer = Lambda(lambda x: x[:, 4, :, :])
                getim5layer = Lambda(lambda x: x[:, 5, :, :])
                getim6layer = Lambda(lambda x: x[:, 6, :, :])
                getim7layer = Lambda(lambda x: x[:, 7, :, :])
                getim8layer = Lambda(lambda x: x[:, 8, :, :])
                getim9layer = Lambda(lambda x: x[:, 9, :, :])
                getim10layer = Lambda(lambda x: x[:, 10, :, :])
                getim11layer = Lambda(lambda x: x[:, 11, :, :])
                getim12layer = Lambda(lambda x: x[:, 12, :, :])
                getim13layer = Lambda(lambda x: x[:, 13, :, :])
                getim14layer = Lambda(lambda x: x[:, 14, :, :])
                getim15layer = Lambda(lambda x: x[:, 15, :, :])
                getim16layer = Lambda(lambda x: x[:, 16, :, :])
                getim17layer = Lambda(lambda x: x[:, 17, :, :])
                getim18layer = Lambda(lambda x: x[:, 18, :, :])
                getim19layer = Lambda(lambda x: x[:, 19, :, :])
                getim20layer = Lambda(lambda x: x[:, 20, :, :])
                getim21layer = Lambda(lambda x: x[:, 21, :, :])
                getim22layer = Lambda(lambda x: x[:, 22, :, :])
                getim23layer = Lambda(lambda x: x[:, 23, :, :])
                getim24layer = Lambda(lambda x: x[:, 24, :, :])
                getim25layer = Lambda(lambda x: x[:, 25, :, :])
                getim26layer = Lambda(lambda x: x[:, 26, :, :])
                getim27layer = Lambda(lambda x: x[:, 27, :, :])
                getim28layer = Lambda(lambda x: x[:, 28, :, :])
                getim29layer = Lambda(lambda x: x[:, 29, :, :])
                getim30layer = Lambda(lambda x: x[:, 30, :, :])
                getim31layer = Lambda(lambda x: x[:, 31, :, :])
                if i == 1:
                    outputimsplit = getim1layer(image_dropout)
                elif i == 2:
                    outputimsplit = getim2layer(image_dropout)
                elif i==3:
                    outputimsplit = getim3layer(image_dropout)
                elif i == 4:
                    outputimsplit = getim4layer(image_dropout)
                elif i==5:
                    outputimsplit = getim5layer(image_dropout)
                elif i == 6:
                    outputimsplit = getim6layer(image_dropout)
                elif i==7:
                    outputimsplit = getim7layer(image_dropout)
                elif i == 8:
                    outputimsplit = getim8layer(image_dropout)
                elif i==9:
                    outputimsplit = getim9layer(image_dropout)
                elif i == 10:
                    outputimsplit = getim10layer(image_dropout)
                elif i==11:
                    outputimsplit = getim11layer(image_dropout)
                elif i == 12:
                    outputimsplit = getim12layer(image_dropout)
                elif i==13:
                    outputimsplit = getim13layer(image_dropout)
                elif i == 14:
                    outputimsplit = getim14layer(image_dropout)
                elif i==15:
                    outputimsplit = getim15layer(image_dropout)
                elif i == 16:
                    outputimsplit = getim16layer(image_dropout)
                elif i==17:
                    outputimsplit = getim17layer(image_dropout)
                elif i == 18:
                    outputimsplit = getim18layer(image_dropout)
                elif i==19:
                    outputimsplit = getim19layer(image_dropout)
                elif i == 20:
                    outputimsplit = getim20layer(image_dropout)
                elif i==21:
                    outputimsplit = getim21layer(image_dropout)
                elif i == 22:
                    outputimsplit = getim22layer(image_dropout)
                elif i==23:
                    outputimsplit = getim23layer(image_dropout)
                elif i == 24:
                    outputimsplit = getim24layer(image_dropout)
                elif i==25:
                    outputimsplit = getim25layer(image_dropout)
                elif i == 26:
                    outputimsplit = getim26layer(image_dropout)
                elif i==27:
                    outputimsplit = getim27layer(image_dropout)
                elif i == 28:
                    outputimsplit = getim28layer(image_dropout)
                elif i==29:
                    outputimsplit = getim29layer(image_dropout)
                elif i == 30:
                    outputimsplit = getim30layer(image_dropout)
                else:
                    outputimsplit = getim31layer(image_dropout)
                per_out = Permute((2, 1))(outputimsplit)
                per_out1 = Dense(units=1,
                                   kernel_regularizer=l2(regularizer))(per_out)
                dim_change = Dense(units=128,
                                   kernel_regularizer=l2(regularizer))(lstm_out)
                dim_change2 = Permute((2, 1))(dim_change)
                #per_out2 = Reshape((196, 1))(dim_change)
                attendout3 = Multiply()([per_out1, dim_change2])
                pre_merge = Reshape((1, embedding_size))(attendout3)
                gettx1layer = Lambda(lambda x: x[:, 1, :])
                gettx2layer = Lambda(lambda x: x[:, 2, :])
                gettx3layer = Lambda(lambda x: x[:, 3, :])
                gettx4layer = Lambda(lambda x: x[:, 4, :])
                gettx5layer = Lambda(lambda x: x[:, 5, :])
                gettx6layer = Lambda(lambda x: x[:, 6, :])
                gettx7layer = Lambda(lambda x: x[:, 7, :])
                gettx8layer = Lambda(lambda x: x[:, 8, :])
                gettx9layer = Lambda(lambda x: x[:, 9, :])
                gettx10layer = Lambda(lambda x: x[:, 10, :])
                gettx11layer = Lambda(lambda x: x[:, 11, :])
                gettx12layer = Lambda(lambda x: x[:, 12, :])
                gettx13layer = Lambda(lambda x: x[:, 13, :])
                gettx14layer = Lambda(lambda x: x[:, 14, :])
                gettx15layer = Lambda(lambda x: x[:, 15, :])
                gettx16layer = Lambda(lambda x: x[:, 16, :])
                gettx17layer = Lambda(lambda x: x[:, 17, :])
                gettx18layer = Lambda(lambda x: x[:, 18, :])
                gettx19layer = Lambda(lambda x: x[:, 19, :])
                gettx20layer = Lambda(lambda x: x[:, 20, :])
                gettx21layer = Lambda(lambda x: x[:, 21, :])
                gettx22layer = Lambda(lambda x: x[:, 22, :])
                gettx23layer = Lambda(lambda x: x[:, 23, :])
                gettx24layer = Lambda(lambda x: x[:, 24, :])
                gettx25layer = Lambda(lambda x: x[:, 25, :])
                gettx26layer = Lambda(lambda x: x[:, 26, :])
                gettx27layer = Lambda(lambda x: x[:, 27, :])
                gettx28layer = Lambda(lambda x: x[:, 28, :])
                gettx29layer = Lambda(lambda x: x[:, 29, :])
                gettx30layer = Lambda(lambda x: x[:, 30, :])
                gettx31layer = Lambda(lambda x: x[:, 31, :])
                if i == 1:
                    outputtxsplit = gettx1layer(image_dropout)
                elif i == 2:
                    outputtxsplit = gettx2layer(image_dropout)
                elif i==3:
                    outputtxsplit = gettx3layer(image_dropout)
                elif i == 4:
                    outputtxsplit = gettx4layer(image_dropout)
                elif i==5:
                    outputtxsplit = gettx5layer(image_dropout)
                elif i == 6:
                    outputtxsplit = gettx6layer(image_dropout)
                elif i==7:
                    outputtxsplit = gettx7layer(image_dropout)
                elif i == 8:
                    outputtxsplit = gettx8layer(image_dropout)
                elif i==9:
                    outputtxsplit = gettx9layer(image_dropout)
                elif i == 10:
                    outputtxsplit = gettx10layer(image_dropout)
                elif i==11:
                    outputtxsplit = gettx11layer(image_dropout)
                elif i == 12:
                    outputtxsplit = gettx12layer(image_dropout)
                elif i==13:
                    outputtxsplit = gettx13layer(image_dropout)
                elif i == 14:
                    outputtxsplit = gettx14layer(image_dropout)
                elif i==15:
                    outputtxsplit = gettx15layer(image_dropout)
                elif i == 16:
                    outputtxsplit = gettx16layer(image_dropout)
                elif i==17:
                    outputtxsplit = gettx17layer(image_dropout)
                elif i == 18:
                    outputtxsplit = gettx18layer(image_dropout)
                elif i==19:
                    outputtxsplit = gettx19layer(image_dropout)
                elif i == 20:
                    outputtxsplit = gettx20layer(image_dropout)
                elif i==21:
                    outputtxsplit = gettx21layer(image_dropout)
                elif i == 22:
                    outputtxsplit = gettx22layer(image_dropout)
                elif i==23:
                    outputtxsplit = gettx23layer(image_dropout)
                elif i == 24:
                    outputtxsplit = gettx24layer(image_dropout)
                elif i==25:
                    outputtxsplit = gettx25layer(image_dropout)
                elif i == 26:
                    outputtxsplit = gettx26layer(image_dropout)
                elif i==27:
                    outputtxsplit = gettx27layer(image_dropout)
                elif i == 28:
                    outputtxsplit = gettx28layer(image_dropout)
                elif i==29:
                    outputtxsplit = gettx29layer(image_dropout)
                elif i == 30:
                    outputtxsplit = gettx30layer(image_dropout)
                else:
                    outputtxsplit = gettx31layer(image_dropout)
                shape_im = Permute((2, 1))(outputtxsplit)
                dim_change_im = Dense(units=1,
                                    kernel_regularizer=l2(regularizer))(shape_im)
                pre_merge_txt = Permute((2, 1))(dim_change_im)
                #pre_merge_txt = Reshape((1, embedding_size))(outputtxsplit)
                recurrent_inputs = [pre_merge_txt, pre_merge]
                merged_input = Add()(recurrent_inputs)
                #merged_input = Reshape((1, embedding_size), name='reshape3')(merged_input_temp)
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
    getoutbelayer = Lambda(lambda x: x[:, 1:, :])
    output_be = Reshape((max_token_length+1, embedding_size))(lstm_out_final)
    output_bee = getoutbelayer(output_be)
    output = TimeDistributed(Dense(units=vocabulary_size,
                                    kernel_regularizer=l2(regularizer),
                                    activation='softmax'),
                                    name='output')(output_bee)

    inputs = [text_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = NIC(16, 1024)
    plot_model(model, '../images/NIC.png')
