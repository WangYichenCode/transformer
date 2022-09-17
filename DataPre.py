"""
对数据进行预处理
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
# if __name__ == "__main__":
# original_data = pd.read_excel("D:\\工业数据集\\整理后数据\\无标签数据11.xlsx")
# original_data.to_pickle("D:\\工业数据集\\整理后数据\\无标签数据11.pkl")
# quality_data = pd.read_excel("D:\\工业数据集\\整理后数据\\柴油95%回收温度11.xlsx")
# quality_data.to_pickle("D:\\工业数据集\\整理后数据\\柴油95%回收温度11.pkl")
#
# # original_data['时间'] = pd.to_datetime(original_data['时间'])  # 读取原来的时间格式并且进行转换
# # quality_data['时间'] = pd.to_datetime(quality_data['时间'])
# # original_data.reset_index(drop=True, inplace=True)
# # original_data.to_excel('D:\\工业数据集\\整理后数据\\无标签数据11.xlsx')  # 保存为excel数据
# # quality_data.to_excel('D:\\工业数据集\\整理后数据\\柴油95%回收温度11.xlsx')


def shift_window(ori_data, qua_data, time_step):
    frequency = ori_data['时间'][1] - ori_data['时间'][0]
    qua_data = qua_data.iloc[:-1, :]
    sample = pd.DataFrame()
    unlabeled_data = np.zeros([qua_data.shape[0], time_step + 1, ori_data.shape[1] - 1])
    for i, quality_time in enumerate(qua_data['时间']):
        tmp_matrix = ori_data[ori_data['时间'].between((quality_time - frequency * time_step), quality_time)]  # 关键行
        sample = pd.concat([sample, tmp_matrix], axis=0)
        unlabeled_data[i, :, :] = tmp_matrix.iloc[:, 1:].values
    return qua_data["柴油95%回收温度"].values, unlabeled_data


def train_test_split(x, y, scale_method='none', shuffle=True, train_sample_number=1200):
    # 使用随机种子划分训练集测试集
    if scale_method == 'std':
        normal_x = StandardScaler().fit_transform(x.reshape(-1, x.shape[2]))
        normal_y = StandardScaler().fit_transform(y.reshape(-1, 1))
    elif scale_method == 'minmax':
        normal_x = MinMaxScaler().fit_transform(x.reshape(-1, x.shape[2]))
        normal_y = MinMaxScaler().fit_transform(y.reshape(-1, 1))
    elif scale_method =='none':
        normal_x = x
        normal_y = y.reshape(-1, 1)
    normal_x = normal_x.reshape(-1, x.shape[1], x.shape[2])
    if shuffle == True:
        randomnumber = list(np.random.RandomState(seed=2022).permutation(x.shape[0]))
    else:
        randomnumber = list(range(x.shape[0]))
    # x = x.reshape(-1, )
    shuffled_x = normal_x[randomnumber, :]
    shuffled_y = normal_y[randomnumber, :]
    train_x = shuffled_x[:train_sample_number, :, :]
    test_x = shuffled_x[train_sample_number:, :, :]
    train_y = shuffled_y[:train_sample_number]
    test_y = shuffled_y[train_sample_number:]
    return train_x, train_y, test_x, test_y


def impute(x, strategy='mean'):  # 缺失值填补
    x_2d = x.reshape(-1, x.shape[2])
    imp = SimpleImputer(strategy=strategy, missing_values=0)
    for i in range(x.shape[1] - 1):
        x_2d.iloc[:, i+1] = imp.fit_transform(x_2d.iloc[:, i+1])
    return x.reshape(-1, x.shape[1], x.shape[2])


if __name__ == "__main__":
    original_data = pd.read_pickle('D:\\工业数据集\\整理后数据\\无标签数据11.pkl')
    quality_data = pd.read_pickle('D:\\工业数据集\\整理后数据\\柴油95%回收温度11.pkl')
    label, shift_data = shift_window(original_data, quality_data, 20)
    # save data
    data_save_path = "./data_load/"
    np.save(data_save_path + "x_data_20.npy", shift_data)
    np.save(data_save_path + "label_data_20.npy", label)
    print("data pre is finished")
