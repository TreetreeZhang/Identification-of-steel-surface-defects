import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish 激活函数：一种光滑的非单调激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 多层感知机 (MLP)：包含两个全连接层和Mish激活函数
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=Mish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # 确保 hidden_features 是一个整数
        hidden_features = hidden_features or int(in_features // 4)
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout 防止过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 卷积层：使用Mish作为激活函数，并支持BN + 激活函数的组合
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False,
                 act_layer=Mish):
        super().__init__()
        self.bn_acti = bn_acti
        # 卷积层
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            # 使用BN和Mish激活函数
            self.bn_mish = BN_Mish(nOut, act_layer=act_layer)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_mish(output)
        return output

# BatchNorm + Mish 组合模块
class BN_Mish(nn.Module):
    def __init__(self, nIn, act_layer=Mish):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = act_layer()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output

# 深度可分离卷积：减少计算量并提高卷积层的效率
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, act_layer=Mish):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=nin, bias=bias)
        # 逐点卷积
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        # 批归一化
        self.bn = nn.BatchNorm2d(nout)
        # 激活函数
        self.acti = act_layer()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.acti(x)
        return x

# 下采样模块：用于减少特征图的空间尺寸
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_mish = BN_Mish(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_mish(output)
        return output

# 上采样模块：用于恢复特征图的空间尺寸
class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

# Squeeze-and-Excitation 块用于轻量级的通道注意力
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):  # 略微减少reduction ratio以增加参数
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 简化后的注意力机制
class SimplifiedEffAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):  # 增加head数量
        super(SimplifiedEffAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# 轻量级Transformer块
class LiteTransBlock(nn.Module):
    def __init__(self, dim=224, num_heads=4, mlp_ratio=2., qkv_bias=False, drop=0., attn_drop=0.):  # 略微增加维度和mlp ratio
        super(LiteTransBlock, self).__init__()
        self.attention = SimplifiedEffAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 修改后的主网络架构，增加了适度的参数量
class self_net(nn.Module):
    def __init__(self, classes=4):
        super(self_net, self).__init__()

        # 初始卷积层，增加了通道数
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        # Squeeze-and-Excitation 块
        self.se_block1 = SEBlock(64)

        # 下采样块
        self.downsample_1 = DownSamplingBlock(64, 128)
        self.se_block2 = SEBlock(128)
        self.downsample_2 = DownSamplingBlock(128, 224)

        # 轻量级Transformer块
        self.transformer_block = LiteTransBlock(dim=224)

        # 额外的优化卷积块
        self.extra_block = nn.Sequential(
            DepthwiseSeparableConv(224, 224, kernel_size=3, stride=1, padding=1),
            SEBlock(224)
        )

        # 上采样和合并操作
        self.upsample_1 = UpsampleingBlock(224, 128)
        self.se_block3 = SEBlock(128)
        self.upsample_2 = UpsampleingBlock(128, 64)
        self.se_block4 = SEBlock(64)
        self.upsample_3 = UpsampleingBlock(64, 32)

        # 最终分类卷积层
        self.classifier = nn.Conv2d(32, classes, kernel_size=1, bias=False)

    def forward(self, input):
        input = F.interpolate(input, size=(208, 208), mode='bilinear', align_corners=False)
        x = self.init_conv(input)
        x = self.se_block1(x)

        # 下采样过程
        x = self.downsample_1(x)
        x = self.se_block2(x)
        x = self.downsample_2(x)

        # Transformer-like 块
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 转换为Transformer输入格式
        x = self.transformer_block(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # 重新调整为2D形状

        # 额外的优化卷积块
        x = self.extra_block(x)

        # 上采样和合并过程
        x = self.upsample_1(x)
        x = self.se_block3(x)
        x = self.upsample_2(x)
        x = self.se_block4(x)
        x = self.upsample_3(x)

        # 最终分类层
        out = self.classifier(x)
        out = F.interpolate(out, size=(200, 200), mode='bilinear', align_corners=False)
        return out











