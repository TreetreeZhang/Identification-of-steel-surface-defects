import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish 激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# ECA 注意力机制替代 SEBlock
class ECAAttention(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 多层感知机 (MLP) 模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=Mish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 1.5)  # 稍微提升隐藏层大小
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 简化的 Efficient Attention 模块
class SimplifiedEffAttention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.):  # 使用6个heads
        super(SimplifiedEffAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        assert C % self.num_heads == 0, f"Embedding dimension C={C} must be divisible by num_heads={self.num_heads}"
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# 轻量化 Transformer 块
class LiteTransformerBlock(nn.Module):
    def __init__(self, dim=168, num_heads=6, mlp_ratio=2.5, qkv_bias=False, drop=0., attn_drop=0.):  # 调整维度为168
        super(LiteTransformerBlock, self).__init__()
        self.attention = SimplifiedEffAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, act_layer=Mish):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.acti = act_layer()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.acti(x)
        return x

# 下采样模块
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = DepthwiseSeparableConv(nIn, nConv, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_mish = ECAAttention(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_mish(output)
        return output

# 上采样模块
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

# 改进后的主网络架构
class self_net(nn.Module):
    def __init__(self, classes=4):
        super(self_net, self).__init__()

        # 初始卷积层，适度增加通道数
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            Mish()
        )

        # ECA Attention 替代 SEBlock
        self.eca_block1 = ECAAttention(96)

        # 下采样块
        self.downsample_1 = DownSamplingBlock(96, 128)
        self.eca_block2 = ECAAttention(128)
        self.downsample_2 = DownSamplingBlock(128, 168)  # 确保维度为168, 可被num_heads=6整除

        # 第一个轻量化 Transformer 块
        self.transformer_block1 = LiteTransformerBlock(dim=168)

        # 第二个轻量化 Transformer 块
        self.transformer_block2 = LiteTransformerBlock(dim=168)

        # 优化卷积块
        self.extra_block = nn.Sequential(
            DepthwiseSeparableConv(168, 168, kernel_size=3, stride=1, padding=1),
            ECAAttention(168)
        )

        # 上采样和合并操作
        self.upsample_1 = UpsampleingBlock(168, 128)
        self.eca_block3 = ECAAttention(128)
        self.upsample_2 = UpsampleingBlock(128, 96)
        self.eca_block4 = ECAAttention(96)
        self.upsample_3 = UpsampleingBlock(96, 48)

        # 最终分类卷积层
        self.classifier = nn.Conv2d(48, classes, kernel_size=1, bias=False)

    def forward(self, input):
        input = F.interpolate(input, size=(208, 208), mode='bilinear', align_corners=False)
        x = self.init_conv(input)
        x = self.eca_block1(x)

        # 下采样过程
        x = self.downsample_1(x)
        x = self.eca_block2(x)
        x = self.downsample_2(x)

        # 第一个 Transformer-like 块
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_block1(x)

        # 第二个 Transformer-like 块
        x = self.transformer_block2(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        # 优化卷积块
        x = self.extra_block(x)

        # 上采样和合并过程
        x = self.upsample_1(x)
        x = self.eca_block3(x)
        x = self.upsample_2(x)
        x = self.eca_block4(x)
        x = self.upsample_3(x)

        # 最终分类层
        out = self.classifier(x)
        out = F.interpolate(out, size=(200, 200), mode='bilinear', align_corners=False)
        return out
