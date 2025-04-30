import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
from utils.Scaler import Scaler
import argparse
from torch.utils.data import TensorDataset, DataLoader
from utils.options import args_parser
from data_process.client_FE import CLIENT_FE


class Client_Data_Loader:
    def __init__(self, file_list, test_battery_name, data_type, args):
        self.args = args
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch_size = args.batch_size
        self.test_battery_name = test_battery_name
        self.file_list = file_list
        self.data_type = data_type
        self.fine_tune_data_ratio = args.fine_tune_data_ratio

    def data_resample(self, data):
        resample_data = []
        for i in range(len(data)):
            cycle_data = data[i]
            V_value, C_value, time_value = cycle_data
            time_value = time_value - time_value[0]
            V_value = np.interp(np.linspace(0, time_value[-1], self.args.num_points), time_value, V_value)
            C_value = np.interp(np.linspace(0, time_value[-1], self.args.num_points), time_value, C_value)
            time_value = np.linspace(0, time_value[-1], self.args.num_points)
            cycle_resample_data = np.vstack((V_value, C_value, time_value))
            resample_data.append(cycle_resample_data)
        resample_data = np.array(resample_data, dtype=np.float32)
        return resample_data

    def load_pkl_data(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            charge_data, cc_data, cv_data, discharge_data, raw_soh = pickle.load(f)[0]
        cc_data = self.data_resample(cc_data)
        cv_data = self.data_resample(cv_data)
        discharge_data = self.data_resample(discharge_data)
        cc_voltage = np.expand_dims(cc_data[:,0,:], 1)
        cv_current = np.expand_dims(cv_data[:,1,:], 1)
        dis_charge_voltage = np.expand_dims(discharge_data[:,0,:], 1)
        input_data = np.concatenate((cc_voltage, cv_current, dis_charge_voltage), axis=1)  # 只使用CC充电的电压数据和CV充电的电流数据作为输入
        raw_soh = np.array(raw_soh, dtype=np.float32)
        soh = raw_soh.reshape(-1, 1)
        scaler = Scaler(input_data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        return data, soh

    def _encapsulation(self, train_x, train_y, test_x, test_y):
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=self.seed)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        return data_dict

    def load_data_Feature(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            charge_data, cc_data, cv_data, discharge_data, raw_soh = pickle.load(f)[0]
        # 提取特征
        client_fe, _ = CLIENT_FE(charge_data, cc_data, cv_data, discharge_data, raw_soh)
        raw_soh = np.array(raw_soh, dtype=np.float32).reshape(-1, 1)
        # 归一化
        scaler = Scaler(client_fe)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        return data, raw_soh

    def create_loader(self, get_type='feature'):
        train_data_x = []
        train_data_y = []
        test_data_x = None
        test_data_y = None
        all_raw_data_x = []
        for file_name in self.file_list:
            battery_name = os.path.basename(file_name).split('.')[0]
            if get_type == 'feature':
                data, soh = self.load_data_Feature(file_name)
            elif get_type == 'raw_data':
                data, soh = self.load_pkl_data(file_name)
            else:
                raise ValueError('get_type should be feature or raw_data')
            all_raw_data_x.append(data)
            if battery_name == self.test_battery_name:
                test_data_x = data
                test_data_y = soh
                print(f'test  battery {battery_name} data shape: {data.shape}--{soh.shape}')
            else:
                train_data_x.append(data)
                train_data_y.append(soh)
                print(f'train battery {battery_name} data shape: {data.shape}--{soh.shape}')
        train_data_x = np.concatenate(train_data_x, axis=0)
        train_data_y = np.concatenate(train_data_y, axis=0)
        if self.fine_tune_data_ratio < 1:
            train_data_x, _, train_data_y, _ = train_test_split(train_data_x, train_data_y, test_size=(1-self.fine_tune_data_ratio), random_state=self.seed)
        all_raw_data_x = np.concatenate(all_raw_data_x, axis=0)
        print(f'[{self.data_type} Dataset]: all train data shape: {train_data_x.shape}--{train_data_y.shape}; test data shape: {test_data_x.shape}--{test_data_y.shape}')
        loader_dict = self._encapsulation(train_data_x, train_data_y, test_data_x, test_data_y)
        return loader_dict, all_raw_data_x


if __name__ == '__main__':
    args = args_parser()
    Battery_file_list = [rf'../data/XJTU/Batch-1/processed_data/battery-{i}.pkl' for i in range(1, 4)]  # XJTU
    my_dataset = Client_Data_Loader(Battery_file_list, 'battery-1', 'XJTU', args)
    loader_dict = my_dataset.create_loader()

