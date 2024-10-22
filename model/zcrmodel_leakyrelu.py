import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 多层感知机 (MLP)：包含两个全连接层和LeakyReLU激活函数
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 使用LeakyReLU激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout 防止过拟合

        # 初始化权重
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 实现相同填充的函数
def Apply_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()

    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]

    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1

    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)

    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left

    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)

    return images


# 提取图像块的函数
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']

    if padding == 'same':
        images = Apply_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(f'不支持的填充类型: {padding}. 仅支持 "same" 或 "valid".')

    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)

    return patches


# 将图像块重新转换回去：用于还原原始图像结构
def reverse_patches(images, out_size, ksizes, strides, padding):
    unfold = torch.nn.Fold(output_size=out_size, kernel_size=ksizes, dilation=1, padding=padding, stride=strides)
    patches = unfold(images)

    return patches


# 高效注意力机制
class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., act_layer=nn.LeakyReLU):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)
            output.append(trans_x)

        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)

        return x


# 卷积层：使用LeakyReLU作为激活函数
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False,
                 act_layer=nn.LeakyReLU):
        super().__init__()
        self.bn_acti = bn_acti
        # 卷积层
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # 初始化权重
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.bn_acti:
            self.bn_leaky_relu = BN_LeakyReLU(nOut, act_layer=act_layer)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_leaky_relu(output)
        return output


# BatchNorm + LeakyReLU 组合模块
class BN_LeakyReLU(nn.Module):
    def __init__(self, nIn, act_layer=nn.LeakyReLU):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = act_layer()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


# 深度可分离卷积：减少计算量并提高卷积层的效率
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, act_layer=nn.LeakyReLU):
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

        # 初始化权重
        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.acti(x)
        return x


# Transformer Block：实现了注意力机制和多层感知机的结合
class TransBlock(nn.Module):
    def __init__(self, n_feat=32, dim=288, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm):
        super(TransBlock, self).__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, act_layer=act_layer)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        x = extract_image_patches(x, ksizes=[3, 3], strides=[1, 1], rates=[1, 1], padding='same')
        x = x.permute(0, 2, 1)
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# 长连接模块：通过分块卷积实现空间上的跨层连接
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        if self.bn_acti:
            self.bn_leaky_relu = BN_LeakyReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)
        if self.bn_acti:
            output = self.bn_leaky_relu(output)
        return output


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
        self.bn_leaky_relu = BN_LeakyReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_leaky_relu(output)
        return output


# 上采样模块：用于恢复特征图的空间尺寸
class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


# 像素注意力模块：对每个像素应用注意力机制
class PA(nn.Module):
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


# 通道注意力模块：通过全局平均池化和最大池化实现通道上的注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块：通过卷积操作在空间维度上实现注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM模块：结合通道注意力和空间注意力
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# 优化的卷积块：结合深度可分离卷积、CBAM注意力机制和残差连接
# OptimizedConvBlock
class OCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, use_attention=True):
        super(OCBlock, self).__init__()
        self.use_residual = (in_channels == out_channels and stride == 1)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           act_layer=nn.LeakyReLU)
        self.bn_leaky_relu = BN_LeakyReLU(out_channels)
        self.attention = CBAM(out_channels) if use_attention else None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn_leaky_relu(out)

        # 加入CBAM注意力机制
        if self.attention is not None:
            out = self.attention(out)

        # 残差连接
        if self.use_residual:
            out = out + residual
        return out


# 动态卷积：动态生成卷积核的权重
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, act_layer=nn.LeakyReLU):
        super(DynamicConv, self).__init__()
        self.dynamic_weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.dynamic_weights, a=math.sqrt(5))

    def forward(self, x):
        weight = self.dynamic_weights
        out = F.conv2d(x, weight, self.bias, stride=1, padding=1)
        return out


# 主网络：集成深度可分离卷积、下采样、上采样、注意力机制、残差连接等模块
class self_net(nn.Module):
    def __init__(self, classes=4):
        super().__init__()

        # 初始化卷积块
        self.init_conv = nn.Sequential(
            DepthwiseSeparableConv(3, 32, 3, 1, padding=1, act_layer=nn.LeakyReLU),  # 深度可分离卷积层
            DepthwiseSeparableConv(32, 32, 3, 1, padding=1, act_layer=nn.LeakyReLU),  # 深度可分离卷积层
            DepthwiseSeparableConv(32, 32, 3, 2, padding=1, act_layer=nn.LeakyReLU),  # 深度可分离卷积层，步长为2
        )

        # 批归一化 + LeakyReLU 激活层
        self.bn_leaky_relu_1 = BN_LeakyReLU(32)  # 第一个 BN + LeakyReLU 层
        self.bn_leaky_relu_2 = BN_LeakyReLU(64)  # 第二个 BN + LeakyReLU 层
        self.bn_leaky_relu_3 = BN_LeakyReLU(128) # 第三个 BN + LeakyReLU 层
        self.bn_leaky_relu_4 = BN_LeakyReLU(32)  # 第四个 BN + LeakyReLU 层
        self.bn_leaky_relu_5 = BN_LeakyReLU(16)  # 第五个 BN + LeakyReLU 层
        self.bn_leaky_relu_6 = BN_LeakyReLU(16)  # 第六个 BN + LeakyReLU 层
        self.bn_leaky_relu_7 = BN_LeakyReLU(16)  # 第七个 BN + LeakyReLU 层

        # 下采样块
        self.downsample_1 = DownSamplingBlock(32, 64)   # 第一次下采样
        self.downsample_2 = DownSamplingBlock(64, 128)  # 第二次下采样
        self.downsample_3 = DownSamplingBlock(128, 32)  # 第三次下采样

        # 优化卷积块
        self.Block_1 = nn.Sequential()
        for i in range(3):
            self.Block_1.add_module(f"OCBlock_1_{i}", OCBlock(64, 64, use_attention=True))  # 第一个优化卷积块

        self.Block_2 = nn.Sequential()
        for i in range(12):
            self.Block_2.add_module(f"OCBlock_2_{i}", OCBlock(128, 128, use_attention=True))  # 第二个优化卷积块

        self.Block_3 = nn.Sequential()
        for i in range(12):
            self.Block_3.add_module(f"OCBlock_3_{i}", OCBlock(32, 32, use_attention=True))  # 第三个优化卷积块

        # Transformer 块
        self.transformer1 = TransBlock(dim=288)  # Transformer 处理块

        # 上采样块和优化卷积块
        self.Block_4 = nn.Sequential()
        for i in range(3):
            self.Block_4.add_module(f"OCBlock_4_{i}", OCBlock(32, 32, use_attention=True))  # 第四个优化卷积块
        self.upsample_1 = UpsampleingBlock(32, 16)  # 第一次上采样
        self.Block_5 = nn.Sequential()
        for i in range(3):
            self.Block_5.add_module(f"OCBlock_5_{i}", OCBlock(16, 16, use_attention=True))  # 第五个优化卷积块
        self.upsample_2 = UpsampleingBlock(16, 16)  # 第二次上采样
        self.Block_6 = nn.Sequential()
        for i in range(3):
            self.Block_6.add_module(f"OCBlock_6_{i}", OCBlock(16, 16, use_attention=True))  # 第六个优化卷积块
        self.upsample_3 = UpsampleingBlock(16, 16)  # 第三次上采样

        # 注意力机制
        self.PA = PA(16)  # 通道注意力模块

        # 长连接（用于跳跃连接）
        self.LC1 = LongConnection(64, 16, 3)   # 第一长连接
        self.LC2 = LongConnection(128, 16, 3)  # 第二长连接
        self.LC3 = LongConnection(32, 32, 3)   # 第三长连接

        # 分类器
        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))  # 最后的分类层


    def forward(self, input):
        input = F.interpolate(input, size=(208, 208), mode='bilinear', align_corners=False)
        output0 = self.init_conv(input)
        output0 = self.bn_leaky_relu_1(output0)

        output1_0 = self.downsample_1(output0)
        output1 = self.Block_1(output1_0)
        output1 = self.bn_leaky_relu_2(output1)

        output2_0 = self.downsample_2(output1)
        output2 = self.Block_2(output2_0)
        output2 = self.bn_leaky_relu_3(output2)

        output3_0 = self.downsample_3(output2)
        output3 = self.Block_3(output3_0)
        output3 = self.bn_leaky_relu_4(output3)

        b, c, h, w = output3.shape
        output4 = self.transformer1(output3)
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)

        output4 = self.Block_4(output4)
        output4 = self.upsample_1(output4 + self.LC3(output3))
        output4 = self.bn_leaky_relu_5(output4)

        output5 = self.Block_5(output4)
        output5 = self.upsample_2(output5 + self.LC2(output2))
        output5 = self.bn_leaky_relu_6(output5)

        output6 = self.Block_6(output5)
        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.PA(output6)
        output6 = self.bn_leaky_relu_7(output6)

        out = F.interpolate(output6, size=(200, 200), mode='bilinear', align_corners=False)
        out = self.classifier(out)
        return out
