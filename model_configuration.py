import warnings
import os
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, GRU
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

class model_module:
    def __init__(self):

        warnings.warn('선택한 모델 구성을 시작합니다.')

    def flatten(self, X):
        '''
        Flatten a 3D array.

        Input
        X            A 3D array for lstm, where the array is sample x timesteps x features.

        Output
        flattened_X  A 2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]
        return (flattened_X)

    def lstm_ae_get_model_layer_1(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = LSTM(n_dimensions, return_sequences=False, name="encoder_01")(inputs)
        decoded = RepeatVector(time_step)(encoded)
        decoded = LSTM(input_dim, return_sequences=True, name='decoder_01')(decoded)
        lstm_autoencoder = Model(inputs, decoded)
        lstm_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        lstm_autoencoder.summary()
        return lstm_autoencoder

    def lstm_ae_get_model_layer_2(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = LSTM(n_dimensions, return_sequences=True, name="encoder_01")(inputs)
        encoded = LSTM(int(n_dimensions/2), return_sequences=False, name="encoder_02")(encoded)
        decoded = RepeatVector(time_step)(encoded)
        decoded = LSTM(int(input_dim/2), return_sequences=True, name='decoder_01')(decoded)
        decoded = LSTM(input_dim, return_sequences=True, name='decoder_02')(decoded)
        lstm_autoencoder = Model(inputs, decoded)
        lstm_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        lstm_autoencoder.summary()
        return lstm_autoencoder

    def lstm_ae_get_model_layer_3(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = LSTM(n_dimensions, return_sequences=True, name="encoder_01")(inputs)
        encoded = LSTM(int(n_dimensions * (2 / 3)), return_sequences=True, name="encoder_02")(encoded)
        encoded = LSTM(int(n_dimensions * (1 / 3)), return_sequences=False, name="encoder_03")(encoded)
        decoded = RepeatVector(time_step)(encoded)
        decoded = LSTM(int(input_dim * (1 / 3)), return_sequences=True, name='decoder_01')(decoded)
        decoded = LSTM(int(input_dim * (2 / 3)), return_sequences=True, name='decoder_02')(decoded)
        decoded = LSTM(input_dim, return_sequences=True, name='decoder_03')(decoded)
        lstm_autoencoder = Model(inputs, decoded)
        lstm_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        lstm_autoencoder.summary()
        return lstm_autoencoder

    def gru_ae_get_model_layer_1(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = GRU(n_dimensions, return_sequences=False, name="encoder_01")(inputs)
        decoded = RepeatVector(time_step)(encoded)
        decoded = GRU(input_dim, return_sequences=True, name='decoder_01')(decoded)
        gru_autoencoder = Model(inputs, decoded)
        gru_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        gru_autoencoder.summary()
        return gru_autoencoder

    def gru_ae_get_model_layer_2(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = GRU(n_dimensions, return_sequences=True, name="encoder_01")(inputs)
        encoded = GRU(int(n_dimensions/2), return_sequences=False, name="encoder_02")(encoded)
        decoded = RepeatVector(time_step)(encoded)
        decoded = GRU(int(input_dim/2), return_sequences=True, name='decoder_01')(decoded)
        decoded = GRU(input_dim, return_sequences=True, name='decoder_02')(decoded)
        gru_autoencoder = Model(inputs, decoded)
        gru_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        gru_autoencoder.summary()
        return gru_autoencoder

    def gru_ae_get_model_layer_3(self, time_step, input_dim, n_dimensions, optimizer, loss, metrics):
        inputs = Input(shape=(time_step, input_dim))
        encoded = GRU(n_dimensions, return_sequences=True, name="encoder_01")(inputs)
        encoded = GRU(int(n_dimensions * (2 / 3)), return_sequences=True, name="encoder_02")(encoded)
        encoded = GRU(int(n_dimensions * (1 / 3)), return_sequences=False, name="encoder_03")(encoded)
        decoded = RepeatVector(time_step)(encoded)
        decoded = GRU(int(input_dim * (1 / 3)), return_sequences=True, name='decoder_01')(decoded)
        decoded = GRU(int(input_dim * (2 / 3)), return_sequences=True, name='decoder_02')(decoded)
        decoded = GRU(input_dim, return_sequences=True, name='decoder_03')(decoded)
        gru_autoencoder = Model(inputs, decoded)
        gru_autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        gru_autoencoder.summary()
        return gru_autoencoder

    def generate_model(self, type, layer, time_step, input_dim, n_dimensions, optimizer='adam', loss='mse', metrics=['acc', 'cosine_proximity']):
        warnings.warn(f'optimizer: {optimizer}, loss: {loss}, metrics: {metrics}는 디폴트로 설정되어 있습니다. 수정이 필요할 경우 지정해주세요.')
        if type == 'LSTM-AE':
            if layer == 1:
                self.model = self.lstm_ae_get_model_layer_1(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            elif layer == 2:
                self.model = self.lstm_ae_get_model_layer_2(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            elif layer == 3:
                self.model = self.lstm_ae_get_model_layer_3(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            else:
                warnings.warn('해당 레이어는 존재하지 않습니다.')
                print('해당 레이어는 존재하지 않습니다.')

        elif type == 'GRU-AE':
            if layer == 1:
                self.model = self.gru_ae_get_model_layer_1(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            elif layer == 2:
                self.model = self.gru_ae_get_model_layer_2(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            elif layer == 3:
                self.model = self.gru_ae_get_model_layer_3(time_step=time_step, input_dim=input_dim, n_dimensions=n_dimensions, optimizer=optimizer, loss=loss, metrics=metrics)
            else:
                warnings.warn('해당 레이어는 존재하지 않습니다.')
                print('해당 레이어는 존재하지 않습니다.')
        else:
            warnings.warn('해당 모델은 존재하지 않습니다.')
        return self.model

    def model_train(self, models, model_name, layer, time_step, batch_size, epoch, train_data, test_data, test_label, train_name, save_interval):
        auc_list = []
        print('model 훈련을 시작합니다.')
        save_directory = f'./{model_name}/{train_name}/{layer}_layer'
        if not os.path.exists(save_directory): os.makedirs(save_directory)
        print(f'train data shape : {np.shape(train_data)}, test data shape : {np.shape(test_data)}, test label shape : {np.shape(test_label)}')
        for _ in range(epoch):
            print(f'훈련 모델 : {train_name}, 현재 반복 횟수: {_ + 1}번')
            history = models.fit(train_data, train_data, batch_size=batch_size, epochs=1)
            valid_x_predictions = models.predict(train_data)
            mse = np.mean(np.power(self.flatten(train_data) - self.flatten(valid_x_predictions), 2), axis=1)
            error_ = pd.DataFrame({'reconstruction_error': mse, })
            temp = error_.describe()
            threshold = (temp.iloc[1].values) + (3 * (temp.iloc[2].values))
            if (_ + 1) % save_interval == 0:
                valid_x_predictions = models.predict(test_data)
                mse = np.mean(np.power(self.flatten(test_data) - self.flatten(valid_x_predictions), 2), axis=1)
                LABELS = ['Train', 'Untrain']
                error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': test_label})
                pred_y = [1 if e > threshold else 0 for e in error_df['Reconstruction_error'].values]
                conf_matrix = metrics.confusion_matrix(error_df['True_class'], pred_y)
                plt.figure(figsize=(7, 7))
                sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Class');
                plt.ylabel('True Class')
                plt.savefig(f'{save_directory}/Confusion_matrix_{train_name}_epoch{_ + 1}_layer{layer}_timestep_{time_step}_batchsize_{batch_size}_threshold_{threshold}.png')
                plt.close()

                false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(error_df['True_class'], error_df['Reconstruction_error'])
                roc_auc = metrics.auc(false_pos_rate, true_pos_rate, )

                plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
                plt.plot([0, 1], [0, 1], linewidth=5)

                plt.xlim([-0.01, 1])
                plt.ylim([0, 1.01])
                plt.legend(loc='lower right')
                plt.title('Receiver operating characteristic curve (ROC)')
                plt.ylabel('True Positive Rate');
                plt.xlabel('False Positive Rate')
                plt.savefig(f'{save_directory}/ROC_{train_name}_epoch{_ + 1}_layer{layer}_timestep_{time_step}_batchsize_{batch_size}_threshold_{threshold}.png')
                plt.close()
                models.save(f'{save_directory}/{train_name}_epoch{_ + 1}_layer{layer}_timestep_{time_step}_batchsize_{batch_size}_threshold_{threshold}_AUC_{round(roc_auc, 3)}.h5')
                auc_list.append(round(roc_auc,3))
        return auc_list



