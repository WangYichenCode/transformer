import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(normalized_shape)
    def forward(self, x, y):
        bn_output = self.bn(self.dropout(y.permute(1, 2, 0)) + x.permute(1, 2, 0))
        return bn_output.permute(2, 0, 1)


class CNNlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNNlayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(
            CNNlayer(in_channels=1, out_channels=8, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNlayer(in_channels=8, out_channels=15, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNlayer(in_channels=15, out_channels=24, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
                    )
        self.fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv(x)
        fc_out = self.fc(conv_out.view(conv_out.size(0), -1))
        return fc_out


class TransCNN(nn.Module):
    def __init__(self, in_feature, dim_model, num_layers, num_heads, seq_length):
        super(TransCNN, self).__init__()
        self.attention = nn.MultiheadAttention(dim_model, num_heads)
        self.add = AddNorm(normalized_shape=(in_feature, ), dropout=0.1)
        self.fnn_layer = nn.Sequential(  # 注意力加残差之后直接卷积
            nn.Linear(in_features=43, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=43),
            nn.Dropout(p=0.3, inplace=False),

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        add_output = self.add(x, attention_output)  # 残差连接+batchnorm
        linear_output = self.fnn_layer(add_output)
        fnn_output = self.add(add_output, linear_output)
        return fnn_output


if __name__ == "__main__":
    trans_model = TransCNN(in_feature=43, dim_model=43, num_layers=2, num_heads=1, seq_length=5).cuda()
    cnn_model = CNN()
    test_x = torch.randn((41, 32, 43)).to(device)
    test_X_cnn = torch.randn((32, 1, 41, 43)).to(device)
    out_trans = trans_model(test_x)
    out_cnn = cnn_model(test_X_cnn)
    print(out_trans.shape)
    print('\n', out_trans)
