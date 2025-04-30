import pickle
import numpy as np
import torch
import os
from utils.Scaler import Scaler
from torch.utils.data import TensorDataset, DataLoader
from utils.options import args_parser
from data_process.client_FE import CLIENT_FE


class GAN_LOADER:
    def __init__(self, file_list, data_type, args):
        self.args = args
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch_size = args.batch_size
        self.file_list = file_list
        self.data_type = data_type

    def data_resample(self, data):
        resample_data = []
        for i in range(len(data)):
            cycle_data = data[i]
            V_value, C_value, time_value = cycle_data
            time_value = time_value - time_value[0]
            # 重采样
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
        # 重采样
        # 由于每个cycle的数据长度不一致，需要将其采样到统一长度
        cc_data = self.data_resample(cc_data)
        cv_data = self.data_resample(cv_data)
        discharge_data = self.data_resample(discharge_data)
        cc_voltage = np.expand_dims(cc_data[:,0,:], 1)
        cv_current = np.expand_dims(cv_data[:,1,:], 1)
        dis_charge_voltage = np.expand_dims(discharge_data[:,0,:], 1)
        # 在新的维度上进行拼接
        input_data = np.concatenate((cc_voltage, cv_current), axis=1)  # 只使用CC充电的电压数据和CV充电的电流数据作为输入
        raw_soh = np.array(raw_soh, dtype=np.float32)
        soh = raw_soh.reshape(-1, 1)

        # 归一化，有三种方式
        scaler = Scaler(input_data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)

        return data, soh

    def load_data_Feature(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            charge_data, cc_data, cv_data, discharge_data, raw_soh = pickle.load(f)[0]
        # 提取特征
        client_fe, _ = CLIENT_FE(charge_data, cc_data, cv_data, discharge_data, raw_soh)
        raw_soh = np.array(raw_soh, dtype=np.float32).reshape(-1, 1)

        # data = client_fe
        # 归一化
        scaler = Scaler(client_fe)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)

        return data, raw_soh


    def _encapsulation(self, raw_data):
        '''
        Encapsulate the numpy.array into DataLoader
        :param train_x: numpy.array
        :param train_y: numpy.array
        :param test_x: numpy.array
        :param test_y: numpy.array
        :return:
        '''
        raw_data = torch.from_numpy(raw_data)
        train_loader = DataLoader(TensorDataset(raw_data), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        return train_loader

    def create_loader(self, get_type='feature'):
        raw_data_x = []
        for file_name in self.file_list:
            battery_name = os.path.basename(file_name).split('.')[0]
            if get_type == 'feature':
                data, soh = self.load_data_Feature(file_name)
            elif get_type == 'resample_data':
                data, soh = self.load_pkl_data(file_name)
            else:
                raise ValueError('get_type must be [feature] or [resample_data]')
            raw_data_x.append(data)
            print(f'train battery {battery_name} data shape: {data.shape}')
        raw_data_x = np.concatenate(raw_data_x, axis=0)
        print(f'[{self.data_type} Dataset]: all train data shape:{raw_data_x.shape}')
        data_loader = self._encapsulation(raw_data_x)
        return data_loader,raw_data_x


if __name__ == '__main__':
    args = args_parser()
    Battery_file_list = [rf'../data/XJTU/Batch-1/processed_data/battery-{i}.pkl' for i in range(1, 4)]  # XJTU
    my_dataset = GAN_LOADER(Battery_file_list, 'TongJi', args)
    loader_dict = my_dataset.create_loader()

