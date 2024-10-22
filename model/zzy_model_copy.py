import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 归一化函数
def normalize(x):
    return x.mul_(2).add_(-1)

# 实现相同填充的函数
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]  # 计算输出的行数
    out_cols = (cols + strides[1] - 1) // strides[1]  # 计算输出的列数
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1  # 计算有效的卷积核大小（行）
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1  # 计算有效的卷积核大小（列）
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)  # 填充行数
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)  # 填充列数
    padding_top = int(padding_rows / 2.)  # 上填充
    padding_left = int(padding_cols / 2.)  # 左填充
    padding_bottom = padding_rows - padding_top  # 下填充
    padding_right = padding_cols - padding_left  # 右填充
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)  # 使用ZeroPad2d进行填充
    return images

# 提取图像块的函数
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    从图像中提取块，并将其放入输出的通道维度
    :param images: 形状为 [batch, channels, in_rows, in_cols] 的4D张量
    :param ksizes: [ksize_rows, ksize_cols] 滑动窗口的大小
    :param strides: [stride_rows, stride_cols] 步长
    :param rates: [dilation_rows, dilation_cols] 膨胀率
    :param padding: 填充类型 ('same' 或 'valid')
    :return: 张量
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(f'不支持的填充类型: {padding}. 仅支持 "same" 或 "valid".')

    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches  # 返回 [N, C*k*k, L]，L 为块的总数

# 将图像块转换回去
def reverse_patches(images, out_size, ksizes, strides, padding):
    """
    从图像中提取块，并放入通道维度中
    :param images: 形状为 [batch, channels, in_rows, in_cols] 的4D张量
    :param ksizes: [ksize_rows, ksize_cols] 滑动窗口的大小
    :param strides: [stride_rows, stride_cols] 步长
    :param padding: 填充类型
    :return: 张量
    """
    unfold = torch.nn.Fold(output_size=out_size, kernel_size=ksizes, dilation=1, padding=padding, stride=strides)
    patches = unfold(images)
    return patches  # 返回形状为 [N, C, H, W] 的 4D 张量，其中 H 和 W 为恢复后的图像的高度和宽度

# 计算均值
def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

# 计算标准差
def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x

# 计算和
def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
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
    
# 长连接模块
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output
    
# 带有批归一化和PReLU的层
class BNPReLU(nn.Module):
    def __init__(self, nIn, act_layer=nn.ReLU):
        super(BNPReLU, self).__init__()
        self.bn = nn.BatchNorm2d(nIn)
        self.acti = act_layer()  # 动态选择激活函数，默认为 ReLU

    def forward(self, x):
        return self.acti(self.bn(x))
    
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 4,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
    
# Shuffle Block（通道重排列）
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''通道重排列: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

# ECA层（Efficient Channel Attention）
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

 # 卷积层
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

# DAB模块
class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)
        return output

# 高效注意力机制
class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)  # 输入降维
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)  # 查询、键和值
        self.proj = nn.Linear(dim // 2, dim)  # 投影层
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 将查询、键和值拆分为多个部分进行并行计算
        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力权重
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)
            output.append(trans_x)

        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x

class TransBlock(nn.Module):
    def __init__(self, n_feat=32, dim=288, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(TransBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        hidden_features = int(dim * mlp_ratio)

        # Attention block with dimension 288
        self.atten = EffAttention(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(self.dim)  # `self.dim` should be 288 now
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(self.dim)

        # 升维线性层：将 36 升到 288
        self.linear_proj = nn.Linear(36, 288)  # 升维操作

    def forward(self, x):
        B = x.shape[0]

        # 升维操作：将 36 升到 288
        x = extract_image_patches(x, ksizes=[3, 3], strides=[1, 1], rates=[1, 1], padding='same')
        x = x.permute(0, 2, 1)  # `[B, 40000, 36]`
        x = self.linear_proj(x)  # 升维到 `[B, 40000, 288]`
        print(f"Input shape after linear projection: {x.shape}")  # 输出调试信息

        # Transformer forward pass
        x = x + self.atten(self.norm1(x))  # norm1 should have `normalized_shape` of 288
        x = x + self.mlp(self.norm2(x))  # norm2 should also have `normalized_shape` of 288
        return x


    
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

class DABModuleOptimized(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3, act_layer=nn.ReLU):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn, act_layer=act_layer)
        self.conv1x1_in = nn.Conv2d(nIn, nIn // 2, 1, 1, padding=0, bias=False)

        # 使用空洞卷积
        self.dconv3x1 = nn.Conv2d(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(d, 0), dilation=(d, 1), bias=False)
        self.dconv1x3 = nn.Conv2d(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, d), dilation=(1, d), bias=False)

        self.bn_relu_2 = BNPReLU(nIn // 2, act_layer=act_layer)
        self.conv1x1_out = nn.Conv2d(nIn // 2, nIn, 1, 1, padding=0, bias=False)
        self.shuffle = ShuffleBlock(nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)

        br = self.dconv3x1(output)
        br = self.dconv1x3(br)

        output = output + br
        output = self.bn_relu_2(output)
        output = self.conv1x1_out(output)
        output = self.shuffle(output + input)

        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"ChannelAttention input shape: {x.shape}")  # Debugging line to check input shape
        avg_out = self.fc(self.avg_pool(x))  # Ensure input here has correct shape
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 定义CBAM中的空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# 最终的CBAM模块，包含通道注意力和空间注意力
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class self_net(nn.Module):
    def __init__(self, classes=4):
        super().__init__()

        # 使用简化的UNet编码器
        self.unet_encoder = UNet(in_channels=3, num_classes=classes, base_c=16)  # 减少通道数

        # 1x1 卷积，用于通道转换
        self.conv_proj = nn.Conv2d(4, 16, kernel_size=1)  # 将 4 通道转换为 16 通道

        # 简化卷积块 4
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 上采样和卷积
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 最后的分类器，输出结果
        self.classifier = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, input):
        # 使用嵌入的UNet编码器编码部分
        unet_output = self.unet_encoder(input)

        # 升维操作：将 UNet 输出的 4 通道转换为 16 通道
        unet_output = self.conv_proj(unet_output)

        # 简单卷积块 + 上采样
        output4 = self.conv_block_4(unet_output)
        output4 = self.upsample_1(output4)

        output5 = self.conv_block_5(output4)
        output5 = self.upsample_2(output5)

        output6 = self.conv_block_6(output5)
        output6 = self.upsample_3(output6)

        # 最终输出分类
        out = self.classifier(output6)
        out = F.interpolate(out, size=(200, 200), mode='bilinear', align_corners=False)
        return out