import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_data_loader, loss_function, optimizer):
    model.train()
    train_epoch_loss = 0
    for i,  (x, label) in enumerate(train_data_loader):
        x = x.to(device)
        label = label.to(device)
        y_pred = model(x.view(x.shape[1], x.shape[0], x.shape[2]))
        y_pred = y_pred.squeeze(0).to(device)
        y_pred, label = y_pred.squeeze(1), label.squeeze(1)
        loss = loss_function(y_pred, label)
        train_epoch_loss += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, train_epoch_loss, i


# 计算r2, rmse
def compute_result(y_true, y_predict):  # 计算误差，真实输出值和预测值之间的误差
    y_predict = np.array(y_predict.cpu()).reshape(-1, 1)  # 将list转为array并且reshape，改变数据形式
    y_true = np.array(y_true.cpu()).reshape(-1, 1)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_predict))  # 调用包
    r2_score = metrics.r2_score(y_true, y_predict)
    return np.around(rmse, 4), np.around(r2_score, 4)  # 对浮点数进行保留四位


# model test
def evaluate(model, test_x, test_label, loss_function):
    test_x = test_x.view(test_x.shape[1], test_x.shape[0], test_x.shape[2]).to(device)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_pred = test_pred.squeeze(0)
        test_loss = loss_function(test_pred, test_label.to(device))
    return test_loss, test_pred


# 画图
def draw_plt(y_tru, y_predict, saver_path):
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(range(len(y_tru)), y_tru, color="r", label="y_true", marker='o',  linewidth=1.0, markersize=3)
    plt.plot(range(len(y_predict)), y_predict, color="b", label="y_pred", marker='o', linewidth=1.0, markersize=3)
    plt.title("95% temperature predict")
    plt.xlabel("nums_samples", fontsize=20)
    plt.ylabel("result of predict", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('{}pred_result.png'.format(saver_path))  # 图片保存
    plt.legend(loc="upper right", fontsize=15)
    plt.show()


def loss_plt(train_loss, test_loss, saver_path):
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(torch.tensor(train_loss, device='cpu'), color='r', label='train loss',
             marker='o',  linewidth=1.0, markersize=3)  # 将列表转为CPU上面计算
    plt.plot(torch.tensor(test_loss, device='cpu'), color='b', label='test loss',
             marker='o', linewidth=1.0, markersize=3)
    plt.title("loss")
    plt.xlabel("epochs", fontsize=20)
    plt.ylabel("loss curve", fontsize=20)
    plt.legend(loc="upper right", fontsize=15)
    plt.savefig('{}loss_curve.png'.format(saver_path))  # 图片保存
    plt.show()


# 保存模型和超参数
def save_file(file_path, model, params_save, r2, rmse):
    mid = str(params_save)
    f = open(file_path + 'params.txt', 'w')
    f.writelines(mid)  # 写入参数
    np.savetxt(file_path + 'r2.txt', r2)
    np.savetxt(file_path + 'rmse.txt', rmse)
    torch.save(model, file_path + 'transformer_model.pt')  # 模型的加载与保存 torch.load
