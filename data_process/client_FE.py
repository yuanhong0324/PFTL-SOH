import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis, pearsonr, spearmanr, kendalltau
from scipy import interpolate

plt.rcParams['font.sans-serif'] = ['Kaiti']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def Statistical_Features(data, index):
    '''
    计算测量数据的统计特征
    :return:
    '''
    cycles_features = []
    for cycle in range(len(data)):
        cycle_data = data[cycle]
        cycle_data = cycle_data[index, :]  # 只计算需要数据的统计特征
        # ind = np.where(np.logical_and(cycle_data[0] >= interval[0], cycle_data[0] <= interval[1]))[0]
        # cycle_data = cycle_data[0][ind]
        # 计算均值
        mean_values = np.mean(cycle_data)
        # 计算方差
        variance_values = np.var(cycle_data)
        # # # 计算偏度
        # skewness_values = skew(cycle_data)
        # # 计算峰度
        # kurtosis_values = kurtosis(cycle_data)
        cycles_features.append([mean_values, variance_values])
        # cycles_features.append([mean_values, variance_values, skewness_values, kurtosis_values])
    cycles_features = np.array(cycles_features, dtype=np.float32)  # [N,C]
    # print(cycles_features.shape)
    return cycles_features

def Calculate_correlation_coefficient(features, representation):
    correlation_coefficients = []
    for i in range(features.shape[1]):
        feature_i = features[:, i]
        correlation_coefficient, _ = pearsonr(feature_i, representation)
        correlation_coefficients.append(correlation_coefficient)
    correlation_coefficients = np.array(correlation_coefficients, dtype=np.float32)
    return correlation_coefficients

# 定义指数经验衰减模型函数
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# 定义双指数模型函数
def double_exp_decay(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e

# 定义3阶多项式衰减函数
def polynomial_decay_3rd(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 定义2阶多项式衰减函数
def polynomial_decay_2nd(x, a, b, c):
    return a * x**2 + b * x + c

# 定义一阶多项式衰减函数
def polynomial_decay_1st(x, a, b):
    return a * x + b

def get_fit_R(datas,idx_x,idx_y,fit_type='3rd_linear',is_show=True):
    fit_R = []
    for i in range(len(datas)):
        cycle_data = datas[i]
        x_data = cycle_data[idx_x]
        x_data = x_data - x_data[0]
        y_data = cycle_data[idx_y]
        if fit_type == '3rd_linear':
            p_opt, p_cov = curve_fit(polynomial_decay_3rd, x_data, y_data,maxfev=2000)
        elif fit_type == '2nd_linear':
            p_opt, p_cov = curve_fit(polynomial_decay_2nd, x_data, y_data,maxfev=2000)
        elif fit_type == '1st_linear':
            p_opt, p_cov = curve_fit(polynomial_decay_1st, x_data, y_data,maxfev=2000)
        elif fit_type == 'exp_decay':
            p_opt, p_cov = curve_fit(exp_decay, x_data, y_data,maxfev=2000)
        elif fit_type == 'double_exp_decay':
            p_opt, p_cov = curve_fit(double_exp_decay, x_data, y_data,maxfev=2000)
        else:
            print('fit_type error')
            p_opt = None
        fit_R.append(p_opt)
        if is_show and i == 1:
            plt.figure(figsize=(8, 6))
            plt.scatter(x_data, y_data, label='原始数据点', s=10, marker='x', color='r')
            if fit_type == '3rd_linear':
                plt.plot(x_data, polynomial_decay_3rd(x_data, *p_opt), 'b-', label='3rd_linear模型拟合曲线')
            elif fit_type == '2nd_linear':
                plt.plot(x_data, polynomial_decay_2nd(x_data, *p_opt), 'b-', label='2nd_linear模型拟合曲线')
            elif fit_type == '1st_linear':
                plt.plot(x_data, polynomial_decay_1st(x_data, *p_opt), 'b-', label='1st_linear模型拟合曲线')
            elif fit_type == 'exp_decay':
                plt.plot(x_data, exp_decay(x_data, *p_opt), 'b-', label='exp_decay模型拟合曲线')
            elif fit_type == 'double_exp_decay':
                plt.plot(x_data, double_exp_decay(x_data, *p_opt), 'b-', label='double_exp_decay模型拟合曲线')
            else:
                print('fit_type error')
            plt.xlabel('时间(min)', fontsize=20)
            plt.ylabel('数据值(A)', fontsize=20)
            plt.xticks(fontsize=20)  # 设置X轴刻度标签的字体大小为16
            plt.yticks(fontsize=20)  # 设置Y轴刻度标签的字体大小为16
            plt.legend(fontsize=20)
            plt.show()
    fit_R = np.array(fit_R, dtype=np.float32)
    # print(fit_R.shape)
    return fit_R

def get_time(charge_data, CC_data, CV_data, discharge_data):
    cc_times = []
    discharge_times = []
    cv_times = []
    charge_times = []
    for cycle in range(len(CC_data)):
        cc_data = CC_data[cycle]
        cv_data = CV_data[cycle]
        discharge_data_cycle = discharge_data[cycle]
        charge_data_cycle = charge_data[cycle]
        cc_time = cc_data[2][-1] - cc_data[2][0]
        cv_time = cv_data[2][-1] - cv_data[2][0]
        discharge_time = discharge_data_cycle[2][-1] - discharge_data_cycle[2][0]
        charge_time = charge_data_cycle[2][-1] - charge_data_cycle[2][0]
        cc_times.append(cc_time)
        cv_times.append(cv_time)
        discharge_times.append(discharge_time)
        charge_times.append(charge_time)
    cc_times = np.array(cc_times, dtype=np.float32).reshape(-1, 1)
    cv_times = np.array(cv_times, dtype=np.float32).reshape(-1, 1)
    discharge_times = np.array(discharge_times, dtype=np.float32).reshape(-1, 1)
    charge_times = np.array(charge_times, dtype=np.float32).reshape(-1, 1)
    return cc_times, cv_times, discharge_times, charge_times


def get_partial_data(data, interval, type_idx):
    partial_data = []
    for cycle in range(len(data)):
        cycle_data = data[cycle]
        ind_partial = np.where(np.logical_and(cycle_data[type_idx] >= interval[0], cycle_data[type_idx] <= interval[1]))[0]
        cycle_data = cycle_data[:, ind_partial]
        partial_data.append(cycle_data)
    return partial_data

def calculate_slope(partial_data,x_idx,y_idx):
    slope_list = []
    for cycle in range(len(partial_data)):
        cycle_data = partial_data[cycle]
        cycle_data_c_or_v = cycle_data[y_idx]
        cycle_data_t = cycle_data[x_idx]
        slope = (cycle_data_c_or_v[-1]-cycle_data_c_or_v[0]) / (cycle_data_t[-1] - cycle_data_t[0])
        slope_list.append(slope)
    return np.array(slope_list, dtype=np.float32).reshape(-1, 1)

def draw_ic(x1, y1):
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, 'r-', linewidth=3, label='IC Curve')
    # 添加图例和标签
    plt.legend()
    plt.xlabel('V')
    plt.ylabel('dQ/dV')
    plt.title('IC Curve')
    # 显示图形
    plt.show()

def show_all_ic_dv(ic_dv_list):
    plt.figure(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ic_dv_list)))
    for cycle in range(len(ic_dv_list)):
        cycle_ic_dv = ic_dv_list[cycle]
        plt.plot(cycle_ic_dv[1], cycle_ic_dv[0], 'r-', linewidth=1, label=f'Cycle_{cycle + 1}', color=colors[cycle])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(ic_dv_list)))
    sm.set_array([])  # 设置虚拟数组，使得colorbar能够正确映射颜色
    cbar = plt.colorbar(sm, label='循环数')
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.label.set_fontsize(18)
    plt.show()

def get_IC_curve(datas, step_len=0.01,is_show=True):
    # 计算增量容量曲线，公式为：dQ/dV = I·dt/dV
    ic_list = []
    for cycle in range(len(datas)):
        cycle_data = datas[cycle]
        V,I,t,cap = cycle_data
        # 对t和I进行等电压V间隔重采样
        num_points = int((max(V) - min(V)) / step_len) + 1
        f1 = interpolate.interp1d(V, t, kind='linear')
        V_new = np.linspace(V[0], V[-1], num_points)
        t_new = f1(V_new)
        f2 = interpolate.interp1d(V, I, kind='linear')
        I_new = f2(V_new)
        dt_div_dv = np.gradient(t_new, V_new)
        IC = I_new * dt_div_dv
        IC = savgol_filter(IC, window_length=3, polyorder=1, mode='nearest')
        ic_list.append(np.vstack([IC, V_new]))
    if is_show:
        show_all_ic_dv(ic_list)
    return ic_list

def get_EVDT_ECDT(data, interval, type_idx):
    EVDT_ECDT = []
    for cycle in range(len(data)):
        cycle_data = data[cycle]
        cycle_data_v_c = cycle_data[type_idx]  # 电压或者电流数据
        cycle_data_t = cycle_data[2]  # 时间数据
        inds = np.where(np.logical_and(cycle_data_v_c >= interval[0], cycle_data_v_c <= interval[1]))[0]
        if len(inds) == 0:
            # print(f'No data in {interval} for cycle {cycle}')
            time1 = 0
        else:
            time1 = cycle_data_t[inds[-1]] - cycle_data_t[inds[0]]
            time1 = abs(time1)  # 放电数据减出来时间为负数，因此需要取绝对值
        EVDT_ECDT.append([time1])
    EVDT_ECDT = np.array(EVDT_ECDT, dtype=np.float32)
    return EVDT_ECDT

def get_DV_curve(datas, step_len=0.005,is_show=False):
    # 计算差分电压曲线，公式为：dV/dQ
    DV_list = []
    for cycle in range(len(datas)):
        cycle_data = datas[cycle]
        V,I,t,cap = cycle_data
        # 对cap进行等电压V间隔重采样
        num_points = int((max(V) - min(V)) / step_len) + 1
        V_new = np.linspace(V[0], V[-1], num_points)
        f_cap = interpolate.interp1d(V, cap, kind='linear')
        cap_new = f_cap(V_new)
        DV = np.gradient(V_new, cap_new)
        DV_list.append(np.vstack([DV, cap_new]))
    if is_show:
        show_all_ic_dv(DV_list)
    return DV_list

def IC_statistics(ic_list):
    # 计算IC曲线的均值和标准差
    F_all = []
    for i in range(len(ic_list)):
        ic_data_cycle = ic_list[i][0]
        ic_mean = np.mean(ic_data_cycle)
        ic_std = np.std(ic_data_cycle)
        ic_max = np.max(ic_data_cycle)
        ic_sk = skew(ic_data_cycle)
        ic_kurt = kurtosis(ic_data_cycle)
        ic_last_point = ic_data_cycle[-1]
        F_all.append([ic_mean, ic_std, ic_max,ic_sk,ic_kurt,ic_last_point])
    F_all = np.array(F_all, dtype=np.float32)
    return F_all


def CLIENT_FE(charge_data, cc_data, cv_data, discharge_data, raw_soh):
    statistical_F_charge_1 = Statistical_Features(charge_data, 0)
    statistical_F_charge_2 = Statistical_Features(charge_data, 1)
    statistical_F_cc = Statistical_Features(cc_data, 0)
    statistical_F_cv = Statistical_Features(cv_data, 1)
    statistical_F_discharge = Statistical_Features(discharge_data, 0)  # 只需要均值和方差
    statistical_F = np.concatenate((statistical_F_charge_1, statistical_F_charge_2, statistical_F_discharge,statistical_F_cc,statistical_F_cv), axis=1)

    ECT_CV_02A_04A = get_EVDT_ECDT(cv_data, (0.2, 0.4), 1)  # 0.2A-0.4A的电流差值时间

    EDT_CC_36V_39V = get_EVDT_ECDT(cc_data, (3.6, 3.8), 0)  # 3.6V-3.9V的电压差值时间
    EDT_CC_38V_41V = get_EVDT_ECDT(cc_data, (3.9, 4.1), 0)  # 3.8V-4.1V的电压差值时间
    EDT_CC_33V_36V = get_EVDT_ECDT(cc_data, (3.3, 3.6), 0)  # 3.3V-3.6V的电压差值时间

    EDT_DIS_37V_40V = get_EVDT_ECDT(discharge_data, (3.7, 4.0), 0)  # 3.7V-4.0V的电压差值时间
    EDT_DIS_33V_36V = get_EVDT_ECDT(discharge_data, (3.3, 3.6), 0)  # 3.3V-3.6V的电压差值时间

    ECT_EDT_F = np.concatenate((ECT_CV_02A_04A, EDT_CC_36V_39V, EDT_CC_38V_41V,EDT_CC_33V_36V,EDT_DIS_37V_40V, EDT_DIS_33V_36V), axis=1)

    CCT,CVT,DIST,CC_CV_T = get_time(charge_data,cv_data, cc_data, discharge_data)
    Time_F = np.concatenate((CCT,CVT,DIST,CC_CV_T), axis=1)

    IC_CC = get_IC_curve(cc_data,is_show=False)
    IC_CC_mean_var_max_sk = IC_statistics(IC_CC)  # CC曲线的均值

    IC_discharge = get_IC_curve(discharge_data,is_show=False)
    IC_discharge_mean_var_max_sk = IC_statistics(IC_discharge)

    IC_Feature = np.concatenate((IC_CC_mean_var_max_sk, IC_discharge_mean_var_max_sk), axis=1)
    all_Features = np.concatenate((statistical_F,ECT_EDT_F,Time_F,IC_Feature), axis=1)
    cof_person = Calculate_correlation_coefficient(all_Features, raw_soh)

    return all_Features,cof_person


if __name__ == '__main__':
    Battery_file_list_client_1 = [rf'../data/TongJi/Dataset_3_NCM_NCA_battery/processed_data/CY25-05_2-#{i}.pkl' for i in [1, 2, 3]]  # 1
    Battery_file_list_client_2 = [rf'../data/TongJi/Dataset_3_NCM_NCA_battery/processed_data/CY25-05_4-#{i}.pkl' for i in [1, 2, 3]]
    Battery_file_list_client_3 = [rf'../data/XJTU/Batch-1/processed_data/battery-{i}.pkl' for i in [2, 3, 4]]  # 2
    Battery_file_list_client_4 = [rf'../data/XJTU/Batch-2/processed_data/battery-{i}.pkl' for i in [1, 2, 3]]  # 3

    ALL_BATTERY_FILE_LIST = [Battery_file_list_client_1, Battery_file_list_client_2, Battery_file_list_client_3, Battery_file_list_client_4]  # 选择需要分析的客户端数据集

    for Battery_file_list in ALL_BATTERY_FILE_LIST:
        for battery_file in Battery_file_list:
            with open(battery_file, 'rb') as f:
                charge_data, CC_data, CV_data, discharge_data, capacity = pickle.load(f)[0]
            all_F, _ = CLIENT_FE(charge_data, CC_data, CV_data, discharge_data, capacity)
        print('客户端特征提取完成')
