import datetime
import gc
import os.path
import numpy as np
import torch
import torch.utils.data as Data
import time
import torch.nn as nn
from TransModel import trans_encoder
from DataPre import  train_test_split
from function import train, compute_result, evaluate, draw_plt, loss_plt, save_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./data_load/"
process_data = np.load(data_path + "x_data_40.npy")
quality_data = np.load(data_path + "label_data_40.npy")

# 标准化、归一化
scale = 'minmax'
train_x, train_label, test_x, test_label = train_test_split(process_data, quality_data,scale_method=scale, shuffle=True,
                                                            train_sample_number=1200)
print("data is split already! \n num_steps :{}, num_process_variable :{}"
      "\ntrain data size:{}, test data size:{}"
      .format(train_x.shape[1], train_x.shape[2], train_x.shape, test_x.shape))

train_x_tensor = torch.from_numpy(train_x).float()  # 与原来的数据共享内存
train_label_tensor = torch.from_numpy(train_label).float()
test_x_tensor = torch.from_numpy(test_x).float()
test_label_tensor = torch.from_numpy(test_label).float()
train_data = Data.TensorDataset(train_x_tensor, train_label_tensor)  # 转成Data形式的数据
test_data = Data.TensorDataset(test_x_tensor, test_label_tensor)
# 保存模型
time_format = '%Y-%m-%d-%H-%M-%S'

t = datetime.datetime.now().strftime(time_format)
file_saver = './model_and_params_saver/' + str(t) + '/'
if not os.path.exists(file_saver): os.makedirs(file_saver)
# training parameters
epochs = 30
learning_rate = 0.0001
batch_size = 20
# step_size = 49  # 学习率衰减的epoch数
# gamma = 0.5  # StepLR
# L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。
# weight_decay = 0.5  # L2正则化系数

# model parameters
in_feature = 43
dim_model = 128
num_layers = 1
num_head = 8
seq_length = 41

model = trans_encoder(in_feature=in_feature, dim_model=dim_model,
                      num_layers=num_layers, num_head=num_head, seq_length=seq_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss().to(device)
# learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size)
trainloader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)

param_dict = {'epoch': epochs,
              'learning_rate': learning_rate,
              'batch_size': batch_size,
              'in_feature': in_feature,
              'dim_model': dim_model,
              'num_layers': num_layers,
              'seq_length': seq_length,
              'scale_method': scale,
              }
print("train begin")
train_loss = []
test_loss = []
rmse_tr_array = []
rmse_te_array = []
r2_tr_array = []
r2_te_array = []
start_time = time.time()
for epoch in range(epochs + 1):
    gc.collect()
    model, train_epoch_loss, i = train(model=model, train_data_loader=trainloader,
                                       loss_function=loss_function, optimizer=optimizer)
    test_lo, test_predict = evaluate(model, test_x_tensor, test_label_tensor, loss_function=loss_function)
    train_lo, train_predict = evaluate(model, train_x_tensor, train_label_tensor, loss_function=loss_function)
    train_loss.append((train_epoch_loss / (i + 1)))
    test_loss.append(test_lo)
    RMSE_te, r2_te = compute_result(y_true=test_label_tensor, y_predict=test_predict.detach())  # 测试集上面计算
    RMSE_tr, r2_tr = compute_result(y_true=train_label_tensor, y_predict=train_predict.detach())
    rmse_te_array.append(RMSE_te)
    rmse_tr_array.append(RMSE_tr)
    r2_tr_array.append(r2_tr)
    r2_te_array.append(r2_te)
    print("train epoch: {} train loss:  {:.4f} test_loss:{:.4f} RMSE_test_data: {:.4f} RMSE_train_data {:.4f} "
          " R2_test_data: {:.4f} R2_train_data: {:.4f}\n"
          .format(epoch, train_lo, test_lo, RMSE_te, RMSE_tr, r2_te, r2_tr))
end_time = time.time()
print("train end, using time:{:.2f}s".format(end_time - start_time))

y_predict = test_predict.reshape(-1, 1).cpu()
y_truth = test_label_tensor.reshape(-1, 1).cpu()

draw_plt(y_tru=y_truth, y_predict=y_predict, saver_path=file_saver)
loss_plt(train_loss=train_loss, test_loss=test_loss, saver_path=file_saver)
save_file(file_path=file_saver, model=model, params_save=param_dict, r2=r2_te_array, rmse=rmse_te_array)
