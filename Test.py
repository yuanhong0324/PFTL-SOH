import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from data_process.client_loader import Client_Data_Loader
from utils.options import args_parser
from utils.util import set_seed, eval_metrix
from models.CNN_AT_KAN import CNN_AT_KAN

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

my_colors_4 = ['#000000', '#9192ab', '#7898e1', '#7cd6cf']

my_colors_8 = ['#5E6C82', '#7F8A9B', '#899FB0', '#81B3A9', '#B3C6BB', '#B7CBD5', '#C1DDDB', '#D1DED7']
colors_4 = ['#000000', '#99bcef', '#57f4ca', '#a7fb98', '#d6b585']

def format_func(value, tick_number):
    return f'{int(value * 100)}%'


def predict(model, test_loader, args, weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=args.device))
    model.eval()
    pre_list = []
    label_list = []

    #模型预热
    random_input = torch.randn(128, 32).to(args.device)
    for i in range(2):
        _, _, _, _ = model(random_input, is_visual_attn=True, is_visual_pre=True)

    time_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            time_start = time.time()
            output, attn, _, out_pre = model(data, is_visual_attn=True, is_visual_pre=True)
            time_record = time.time() - time_start
            time_list.append(time_record)
            # print(f"batch {batch_idx} time cost: {time_record:.5f}s")
            pre_list.append(output.cpu().numpy())
            label_list.append(target.cpu().numpy())
        time_pre = np.mean(time_list)
        # print(f"average time cost: {time_pre:.5f}s")
    pre_list = np.concatenate(pre_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    error = eval_metrix(label_list, pre_list)
    # print(f"MAE: {error[0]:.4f}, RMSE: {error[1]:.4f}")
    return error


if __name__ == '__main__':
    # 加载数据
    test_battery_idx = 2
    Scenario_id = 2
    args = args_parser()
    set_seed(args.random_seed)
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cpu')
    Battery_file_list_client_1 = [rf'./data/TongJi/Dataset_3_NCM_NCA_battery/processed_data/CY25-05_2-#{i}.pkl' for i in [1, 2, 3]]  # 1
    Battery_file_list_client_2 = [rf'./data/TongJi/Dataset_3_NCM_NCA_battery/processed_data/CY25-05_4-#{i}.pkl' for i in [1, 2, 3]]
    Battery_file_list_client_3 = [rf'./data/XJTU/Batch-1/processed_data/battery-{i}.pkl' for i in [2, 3, 4]]  # 2
    Battery_file_list_client_4 = [rf'./data/XJTU/Batch-2/processed_data/battery-{i}.pkl' for i in [1, 2, 3]]  #
    ALL_BATTERY_FILE_LIST = [Battery_file_list_client_1, Battery_file_list_client_2, Battery_file_list_client_3, Battery_file_list_client_4]

    battery_name_list = []
    loader_list = []

    for i, battery_file_list in enumerate(ALL_BATTERY_FILE_LIST):
        battery_name_client = [os.path.basename(file_path).split('.')[0] for file_path in battery_file_list]
        battery_name_list.append(battery_name_client)
        loader, raw_data = Client_Data_Loader(battery_file_list, battery_name_client[test_battery_idx - 1],
                                              f'CLIENT{i + 1}', args).create_loader(get_type='feature')
        loader_list.append(loader)

    PFL_method_list = ['Proposed','5_FedPer', '6_FedBN']

    model = CNN_AT_KAN()
    model.to(args.device)
    method = 'Proposed'

    cilent_mean_error_list = []
    print("*" * 20, f"Print Task {test_battery_idx} SOH estimation error", "*" * 20)
    for i, client_id in enumerate(range(1, 5)):
        exp_mean_error_list = []
        for exp_id in range(1, 5):
            if Scenario_id == 2:
                weight_path = f'./save_FTL/Scenario 2/{method}/test_{test_battery_idx}_battery/experiment_{exp_id}/client_{client_id}/best_net.pth'
                if method not in PFL_method_list:
                    weight_path = f'./save_FTL/Scenario 2/{method}/test_{test_battery_idx}_battery/experiment_{exp_id}/best_net.pth'
            else:
                weight_path = f'./save_FTL/{method}/test_{test_battery_idx}_battery/experiment_{exp_id}/client_{client_id}/best_net.pth'
                if method not in PFL_method_list:
                    weight_path = f'./save_FTL/{method}/test_{test_battery_idx}_battery/experiment_{exp_id}/best_net.pth'
            test_loader = loader_list[client_id - 1]['test']
            error_e = predict(model, test_loader, args, weight_path)
            exp_mean_error_list.append(error_e)
        exp_mean_error = np.mean(exp_mean_error_list, axis=0)
        print(f"client {client_id} MAE: {exp_mean_error[0]:.5f}, RMSE: {exp_mean_error[1]:.5f}")
        cilent_mean_error_list.append(exp_mean_error)
    cilent_mean_error_list = np.array(cilent_mean_error_list)
    client_mean_error = np.mean(cilent_mean_error_list, axis=0)
    print(f"client mean MAE: {client_mean_error[0]:.5f}, RMSE: {client_mean_error[1]:.5f}")





