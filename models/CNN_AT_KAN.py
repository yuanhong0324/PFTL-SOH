import torch
from torch import nn
from models.KAN import KANLinear, KAN_PRE


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expansion_ratio=3,
                 apply_transform=True, transform_scale=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        # x = self.fc(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next.mean(dim=1).squeeze()


class EncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_head=8, num_layers=2, FC_dim=128):
        super(EncoderBlock, self).__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                                        dim_feedforward=FC_dim, dropout=0.0,
                                                                        batch_first=True)
                                             for _ in range(num_layers)])
        self.attention_weights = None

        self.re_attention = ReAttention(dim=d_model, num_heads=n_head)

    def forward(self, x):
        self.attention_weights = None  # 重置注意力权重

        for idx, layer in enumerate(self.encoder_layers):
            # 提取注意力权重
            attn_output, attn_output_weights = self.re_attention(x)
            if idx == len(self.encoder_layers) - 1:
                self.attention_weights = attn_output_weights  # 保存最后一层的注意力权重
            # 剩余的前向传播逻辑
            x = layer.norm1(x + layer.dropout1(attn_output))
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(x2))

        return x, self.attention_weights


class CNN_AT_KAN(nn.Module):
    def __init__(self, feature_size=32, input_dim=4, extract_dim=64):
        super(CNN_AT_KAN, self).__init__()
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.extract_dim = extract_dim
        self.layer_0 = nn.Linear(feature_size, self.input_dim * self.extract_dim)

        self.cnn = nn.Sequential(ResBlock(input_channel=input_dim, output_channel=16, stride=1),
                                 ResBlock(input_channel=16, output_channel=32, stride=1),
                                 ResBlock(input_channel=32, output_channel=64, stride=1))

        self.encoder = EncoderBlock(d_model=64, FC_dim=80, num_layers=2)

        self.down_conv = nn.Conv1d(64, 16, 1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.predictor = nn.Sequential(KAN_PRE(layers_hidden=[16 * 16, 16, 1]))

    def forward(self, x, is_MOON=False, is_visual_attn=False, is_visual_pre=False):
        x = self.layer_0(x)
        x = x.view(x.size(0), self.input_dim, self.extract_dim)
        x = self.cnn(x)

        x = self.pool(x)
        x = x.transpose(1, 2)
        x, attn = self.encoder(x)
        mmd_out_2 = x.reshape(x.size(0), -1)

        x = x.transpose(1, 2)
        x = self.down_conv(x)
        x = self.pool(x)
        out_pre = x.reshape(x.size(0), -1)
        out = self.predictor(out_pre)

        if not is_visual_attn:
            attn = attn.view(attn.size(0), -1)
        attn = attn.mean(dim=0).squeeze()

        if is_MOON or is_visual_pre:
            pro = out_pre
            return out, attn, mmd_out_2, pro
        else:
            return out, attn, mmd_out_2


# 对比模型
class CNN_AT_MLP(nn.Module):
    def __init__(self, feature_size=32, input_dim=4, extract_dim=64):
        super(CNN_AT_MLP, self).__init__()
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.extract_dim = extract_dim
        self.layer_0 = nn.Linear(feature_size, self.input_dim * self.extract_dim)

        self.cnn = nn.Sequential(ResBlock(input_channel=input_dim, output_channel=16, stride=1),
                                 ResBlock(input_channel=16, output_channel=32, stride=1),
                                 ResBlock(input_channel=32, output_channel=64, stride=1))

        self.encoder = EncoderBlock(d_model=64, FC_dim=80, num_layers=2)

        self.down_conv = nn.Conv1d(64, 16, 1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.predictor = nn.Sequential(nn.Linear(16 * 16, 16),
                                       nn.ReLU(),
                                       nn.Linear(16, 1))

    def forward(self, x, is_MOON=False):
        x = self.layer_0(x)
        x = x.view(x.size(0), self.input_dim, self.extract_dim)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, attn = self.encoder(x)
        mmd_out_2 = x.reshape(x.size(0), -1)

        x = x.transpose(1, 2)
        x = self.down_conv(x)
        x = self.pool(x)
        out_pre = x.reshape(x.size(0), -1)
        out = self.predictor(out_pre)

        attn = attn.view(attn.size(0), -1)
        attn = attn.mean(dim=0).squeeze()

        if is_MOON:
            pro = out_pre
            return out, attn, mmd_out_2, pro
        else:
            return out, attn, mmd_out_2
class CNN_KAN(nn.Module):
    def __init__(self, feature_size=32, input_dim=4, extract_dim=64):
        super(CNN_KAN, self).__init__()
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.extract_dim = extract_dim
        self.layer_0 = nn.Linear(feature_size, self.input_dim * self.extract_dim)

        self.cnn = nn.Sequential(ResBlock(input_channel=input_dim, output_channel=16, stride=1),
                                 ResBlock(input_channel=16, output_channel=32, stride=2),
                                 ResBlock(input_channel=32, output_channel=64, stride=2))
        self.predictor = nn.Sequential(KAN_PRE(layers_hidden=[64 * 16, 16, 1]))

    def forward(self, x, is_MOON=False):
        x = self.layer_0(x)
        x = x.view(x.size(0), self.input_dim, self.extract_dim)
        x = self.cnn(x)
        out_pre = x.reshape(x.size(0), -1)
        out = self.predictor(out_pre)
        return out, None, None


