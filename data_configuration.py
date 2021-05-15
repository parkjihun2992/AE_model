import os
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

class data_module:
    def __init__(self):
        self.top_path = './DataBase/CNS_db/pkl'
        self.file_list = os.listdir(self.top_path)
        self.selected_para = pd.read_csv('./DataBase/Final_parameter_200825.csv')['0'].tolist() # 파라미터 변경시 수정
        self.scale_value = {'min_value':[], 'max_value':[]}
        self.train_data = {'ab21-01': [], 'ab21-02': [], 'ab20-04': [],
                        'ab15-07': [], 'ab15-08': [], 'ab63-04': [], 'ab63-02': [],
                        'ab21-12': [], 'ab19-02': [], 'ab21-11': [],
                        'ab60-02': [], 'ab23-03': [], 'ab59-02': [], 'ab23-01': [], 'ab23-06': [],
                        'normal': [], 'train_untrain': []}
        self.test_data = []
        self.test_label = {'ab21-01': [], 'ab21-02': [], 'ab20-04': [],
                        'ab15-07': [], 'ab15-08': [], 'ab63-04': [], 'ab63-02': [],
                        'ab21-12': [], 'ab19-02': [], 'ab21-11': [],
                        'ab60-02': [], 'ab23-03': [], 'ab59-02': [], 'ab23-01': [], 'ab23-06': [],
                        'normal': [], 'train_untrain': []}

    def train_test_split(self, skip=True, save=True):
        random.seed(10)
        print('Random seed는 10으로 고정됩니다.')
        while True:
            if skip:
                try:
                    with open('train_test_split_result.pkl', 'rb') as f:  # 사전에 저장된 split 데이터 활용, 이는 정확한 분류를 위함.
                        self.train_test_split = pickle.load(f)
                    print('Skip mode 활성화로 인해 저장된 파일로부터 Train&Test 파일 불러오기를 완료하였습니다.')
                    break
                except:
                    print('저장된 파일이 없기에 Skip mode 비활성화로 수행됩니다.')
                    skip = False
            else:
                db_list_sort = {'ab21-01': [], 'ab21-02': [], 'ab20-01': [], 'ab20-04': [],
                                'ab15-07': [], 'ab15-08': [], 'ab63-04': [], 'ab63-02': [],
                                'ab63-03': [], 'ab21-12': [], 'ab19-02': [], 'ab21-11': [],
                                'ab59-01': [], 'ab80-02': [], 'ab64-03': [], 'ab60-02': [],
                                'ab23-03': [], 'ab59-02': [], 'ab23-01': [], 'ab23-06': [],
                                'normal': []}
                for key in db_list_sort.keys():
                    for name in self.file_list:
                        if name[:7] == key:
                            db_list_sort[key].append(name)
                        elif name[:6] == key:
                            db_list_sort[key].append(name)

                train_test_split = {'ab21-01': {'k': 5, 'train_list': [], 'test_list': []}, 'ab21-02': {'k': 5, 'train_list': [], 'test_list': []}, 'ab20-04': {'k': 3, 'train_list': [], 'test_list': []},
                                    'ab15-07': {'k': 5, 'train_list': [], 'test_list': []}, 'ab15-08': {'k': 5, 'train_list': [], 'test_list': []}, 'ab63-04': {'k': 8, 'train_list': [], 'test_list': []}, 'ab63-02': {'k': 2, 'train_list': [], 'test_list': []},
                                    'ab21-12': {'k': 7, 'train_list': [], 'test_list': []}, 'ab19-02': {'k': 6, 'train_list': [], 'test_list': []}, 'ab21-11': {'k': 5, 'train_list': [], 'test_list': []},
                                    'ab60-02': {'k': 5, 'train_list': [], 'test_list': []}, 'ab23-03': {'k': 5, 'train_list': [], 'test_list': []}, 'ab59-02': {'k': 5, 'train_list': [], 'test_list': []}, 'ab23-01': {'k': 5, 'train_list': [], 'test_list': []},
                                    'ab23-06': {'k': 6, 'train_list': [], 'test_list': []},
                                    'normal': {'k': 5, 'train_list': [], 'test_list': []}, 'train_untrain':{'train_list': [], 'test_list': []}}
                test_file_list = ['ab20-01', 'ab63-03', 'ab59-01', 'ab80-02', 'ab64-03']
                test_temp = []

                for key in train_test_split.keys():
                    if key != 'train_untrain':
                        train_test_split[key]['test_list'].append(random.sample(db_list_sort[key], k=train_test_split[key]['k']))
                        train_test_split[key]['train_list'].append([x for x in db_list_sort[key] if x not in train_test_split[key]['test_list'][0]])
                for key in test_file_list:
                    test_temp.append(db_list_sort[key])
                train_test_split['train_untrain']['test_list'].append(np.concatenate(test_temp))
                self.train_test_split = train_test_split
                if save:
                    with open('train_test_split_result.pkl', 'wb') as f:
                        pickle.dump(self.train_test_split, f)
                    print('요청하신 Train&Test 파일의 구분이 완료되었습니다.')
                break
        return self.train_test_split

    def min_max_value_calculator(self, skip=True, want_save=False, train_scaler=True):
        self.trained_scaler = MinMaxScaler()
        while True:
            if skip:
                try:
                    min_value = pd.read_csv('./min_value.csv')['0']
                    max_value = pd.read_csv('./max_value.csv')['0']
                    if train_scaler:
                        self.trained_scaler.fit([min_value, max_value])
                        print('Skip mode 활성화로 인해 저장된 파일로부터 Scaler가 훈련되었습니다.')
                        break
                except:
                    print('저장된 파일이 없기에 Skip mode 비활성화로 수행됩니다.')
                    skip=False
                    want_save=True
            else:
                min_values, max_values = [], []
                for file_name in self.file_list:
                    with open(f'{self.top_path}/{file_name}', 'rb') as f:
                        db = pickle.load(f)
                    db = db[self.selected_para]
                    each_min = db.min(axis=0)
                    each_max = db.max(axis=0)
                    min_values.append(each_min)
                    max_values.append(each_max)

                self.scale_value['min_value'].append(np.array(min_values).min(axis=0))
                self.scale_value['max_value'].append(np.array(max_values).max(axis=0))
                if want_save:
                    pd.DataFrame(self.scale_value['min_value'][0], index=self.selected_para).to_csv('min_value.csv')
                    pd.DataFrame(self.scale_value['max_value'][0], index=self.selected_para).to_csv('max_value.csv')
                    print('요청하신 Min&Max 값 저장이 완료되었습니다.')
                if train_scaler:
                    self.trained_scaler.fit([self.scale_value['min_value'][0], self.scale_value['max_value'][0]])
                    print('요청하신 Sacler 훈련이 완료되었습니다.')
                break
        return self.trained_scaler

    def generator_data(self, time_step, delete_line=150):
        train_set = {'ab21-01': [], 'ab21-02': [], 'ab20-04': [],
                       'ab15-07': [], 'ab15-08': [], 'ab63-04': [], 'ab63-02': [],
                       'ab21-12': [], 'ab19-02': [], 'ab21-11': [],
                       'ab60-02': [], 'ab23-03': [], 'ab59-02': [], 'ab23-01': [], 'ab23-06': [],
                       'normal': [], 'train_untrain': []}
        test_set = []
        test_set_label = {'ab21-01': [], 'ab21-02': [], 'ab20-04': [],
                            'ab15-07': [], 'ab15-08': [], 'ab63-04': [], 'ab63-02': [],
                            'ab21-12': [], 'ab19-02': [], 'ab21-11': [],
                            'ab60-02': [], 'ab23-03': [], 'ab59-02': [], 'ab23-01': [], 'ab23-06': [],
                            'normal': [], 'train_untrain': []}
        print(f'Time_step: {time_step}에 대한 훈련 데이터 생성을 시작합니다.')
        for train_object in self.train_test_split.keys():
            if train_object == 'train_untrain': pass
            elif train_object == 'normal':
                for temp in self.train_test_split[train_object]['train_list'][0]:
                    with open(f'{self.top_path}/{temp}', 'rb') as f:
                        db = pickle.load(f)
                    train_x = db[self.selected_para]
                    train_x = self.trained_scaler.transform(train_x)
                    train_set[train_object].append([train_x[line:line + time_step] for line in range(len(train_x) - time_step)])
                    train_set['train_untrain'].append([train_x[line:line + time_step] for line in range(len(train_x) - time_step)])
            else:
                # print(train_object)
                for temp in self.train_test_split[train_object]['train_list'][0]:
                    with open(f'{self.top_path}/{temp}', 'rb') as f:
                        db = pickle.load(f)
                    train_x = db[db['KCNTOMS'] >= delete_line]
                    train_x = train_x[self.selected_para]
                    train_x = self.trained_scaler.transform(train_x)
                    train_set[train_object].append([train_x[line:line + time_step] for line in range(len(train_x) - time_step)])
                    train_set['train_untrain'].append([train_x[line:line + time_step] for line in range(len(train_x) - time_step)])
        for train_object in self.train_test_split.keys():
            if train_object != 'train_untrain':
                self.train_data[train_object].append(np.concatenate(train_set[train_object]))
        self.train_data['train_untrain'].append(np.concatenate(train_set['train_untrain']))
        print(f'Time_step: {time_step}에 대한 훈련 데이터 생성이 완료되었습니다.')
        for train_object in self.train_test_split.keys():
            print(f'{train_object}의 훈련 데이터 생성 결과: {np.shape(self.train_data[train_object][0])}')
        print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
        print(f'Time_step: {time_step}에 대한 테스트 데이터 생성을 시작합니다.')
        ab_class_list = []
        for train_object in self.train_test_split.keys():
            # print(train_object)
            for temp in self.train_test_split[train_object]['test_list'][0]:
                ab_class_list.append(temp)
        for ab_test in ab_class_list:
            if ab_test[:6] == 'normal':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    # print(ab_test)
                    db = pickle.load(f)
                test_x = db[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'normal':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab21-01':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab21-01':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab21-02':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab21-02':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab20-04':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab20-04':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab15-07':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab15-07':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab15-08':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab15-08':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab63-04':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab63-04':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab63-02':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab63-02':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab21-12':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab21-12':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab19-02':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab19-02':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab21-11':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab21-11':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab60-02':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab60-02':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab23-03':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab23-03':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab59-02':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab59-02':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab23-01':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab23-01':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            elif ab_test[:7] == 'ab23-06':
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    if train_object == 'ab23-06':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    elif train_object == 'train_untrain':
                        test_set_label[train_object].append(np.zeros(test_shape))
                    else:
                        test_set_label[train_object].append(np.ones(test_shape))
            else:
                with open(f'{self.top_path}/{ab_test}', 'rb') as f:
                    db = pickle.load(f)
                test_x = db[db['KCNTOMS'] >= delete_line]
                test_x = test_x[self.selected_para]
                test_x = self.trained_scaler.transform(test_x)
                test_set.append([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])
                test_shape = np.shape([test_x[line:line + time_step] for line in range(len(test_x) - time_step)])[0]
                for train_object in self.train_test_split.keys():
                    test_set_label[train_object].append(np.ones(test_shape))
        self.test_data.append(np.concatenate(test_set))
        for train_object in self.train_test_split.keys():
            self.test_label[train_object].append(np.concatenate(test_set_label[train_object]))

        print('Train & Test & Label 데이터 구성을 완료하였습니다.')
        print([f'Test 데이터 구조 : {np.shape(self.test_data[0])}, Label 데이터 구조 : {[train_object, np.shape(self.test_label[train_object][0])]}' for train_object in self.train_test_split.keys()])
        return self.train_data, self.test_data, self.test_label


if __name__ == '__main__':
    data_module()