import torch
import torch.nn as nn
"""
测试接口
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trans_encoder(nn.Module):
    def __init__(self, in_feature, dim_model, num_layers, num_head, seq_length):
        super(trans_encoder, self).__init__()
        self.linear_1 = nn.Linear(in_features=in_feature, out_features=dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.linear_2 = nn.Sequential(
            nn.Linear(in_features=(dim_model * seq_length), out_features=int(0.3 * (dim_model * seq_length))),  # 6400
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=int(0.3 * (dim_model * seq_length)), out_features=int(0.1 * (dim_model * seq_length))),
            nn.Linear(in_features=int(0.1 * (dim_model * seq_length)), out_features=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        trans_input = self.linear_1(x)
        encoder_output = self.encoder(trans_input)  # 注意x的格式（seq_length, batch_size, num_features）
        output = self.linear_2(encoder_output.permute(1, 0, 2).reshape(x.shape[1], -1))
        return output


# 格式测试
if __name__ == "__main__":
    trans_model = trans_encoder(in_feature=43, dim_model=128,
                                num_layers=2, num_head=4, seq_length=5).cuda()
    test_x = torch.randn((5, 32, 43)).to(device)
    out = trans_model(test_x)
    print(out.shape)
    print('\n', out)
