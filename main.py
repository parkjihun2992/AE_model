from model_configuration import model_module
from data_configuration import data_module
import numpy as np
import requests
import json
import time


class main:
    def __init__(self):
        model_info = {'model_name': 'GRU-AE', 'layer':3, 'time_step':5, 'batch_size':32, 'epoch':300, 'save_interval':30}
        train_type = 'ab21-01'
        self.data_module=data_module()
        self.model_module = model_module()
        self.data_module.train_test_split() # data module init 포함
        self.data_module.min_max_value_calculator() # data module init 포함
        self.train_data, self.test_data, self.test_label = self.data_module.generator_data(time_step=model_info['time_step'])
        # model 구성시 optimizer, loss, metrics는 디폴트로 설정되어 있음 -> 변경할 경우, 모델 구성시 추가하면 됨. ex) optimizer='adam'
        # train_list = ['ab21-01', 'ab21-02', 'ab20-04', 'ab15-07', 'ab15-08', 'ab63-04', 'ab63-02', 'ab21-12', 'ab19-02', 'ab21-11', 'ab60-02', 'ab23-03', 'ab59-02', 'ab23-01', 'ab23-06', 'normal']
        train_list = ['ab21-02', 'ab23-03', 'ab59-02', 'ab23-01', 'ab23-06', 'normal']
        for i, train_name in enumerate(train_list):
            self.model = self.model_module.generate_model(type=model_info['model_name'], layer=model_info['layer'], time_step=model_info['time_step'], input_dim=np.shape(self.train_data[train_type])[-1], n_dimensions=np.shape(self.train_data[train_type])[-1])
            auc_list = self.model_module.model_train(models=self.model, model_name=model_info['model_name'], layer=model_info['layer'], time_step=model_info['time_step'], batch_size=model_info['batch_size'], epoch=model_info['epoch'], train_data=self.train_data[train_name][0], test_data=self.test_data[0], test_label=self.test_label[train_name][0], train_name=train_name, save_interval=model_info['save_interval'])
            now = time.localtime()
            t = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            self.line_message(text=f'시간 : {t}, {i+1}/{len(train_list)}, {train_name} 훈련 종료 및 auc 결과 : {auc_list}, 가장 높은 auc : {max(auc_list)}')

    def kakao_message(self, text):
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

        # 사용자 토큰
        token = 'dIkZXgL7L6FCJ3Vs-99ljxvTnFbMFaYOTo7miwopdSkAAAF5br5fZw'
        headers = {"Authorization": "Bearer " + token}
        data = {
            "template_object": json.dumps({"object_type": "text",
                                           "text": text,
                                           "link": {
                                               "web_url": "www.naver.com"
                                           }
                                           })
        }

        response = requests.post(url, headers=headers, data=data)
        print(response.status_code)
        if response.json().get('result_code') == 0:
            print('메시지를 성공적으로 보냈습니다.')
        else:
            print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))

    # def line_message(text, figure):
    #     try:
    #
    #         TARGET_URL = 'https://notify-api.line.me/api/notify'
    #         TOKEN = 'sSRDDHWunNjR6mF3TUdBXyv3qRm2Zv3KbvATugNndbl'
    #
    #         with open(figure, 'rb') as file:
    #             # 요청합니다.
    #             response = requests.post(
    #                 TARGET_URL,
    #                 headers={
    #                     'Authorization': 'Bearer ' + TOKEN
    #                 },
    #                 data={
    #                     'message': text,
    #                 },
    #                 files={
    #                     'imageFile': file
    #                 }
    #             )
    #
    #     except Exception as ex:
    #         print(ex)

    def line_message(self, text):
        try:

            TARGET_URL = 'https://notify-api.line.me/api/notify'
            TOKEN = 'sSRDDHWunNjR6mF3TUdBXyv3qRm2Zv3KbvATugNndbl'

            response = requests.post(
                TARGET_URL,
                headers={
                    'Authorization': 'Bearer ' + TOKEN
                },
                data={
                    'message': text,
                }
            )

        except Exception as ex:
            print(ex)

if __name__ == '__main__':
    main()
