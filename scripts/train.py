import matplotlib
matplotlib.use('Agg')  # 设置 matplotlib 使用 'Agg' 后端
import pandas as pd
from utils.dataloader import *
from datetime import datetime
import os
from utils.tools import *
import torch
import torch.nn as nn
import importlib  # 动态导入模块
import matplotlib.pyplot as plt
import numpy as np

class Train():

    """
    Train类，用于定义和控制整个模型训练的流程，包括加载数据、训练模型、评估以及保存最佳模型。
    """

    def __init__(self, model_name='zzy_model2', data_root='datasets_origin_enhance', device='cuda:0', class_num=3, pretrained_model=None, isComplete=True) -> None:
        """
        初始化类的属性，包括设备选择、模型、数据集路径和保存路径。
        
        参数:
        model_name : str, 使用的模型名称
        data_root : str, 数据集的根路径
        device : str, 设备选择
        class_num : int, 类别数
        pretrained_model : str, 预训练模型的路径
        isComplete : bool, 是否直接加载完整模型
        """
        self.device = device
        self.data_root = 'datasets/' + data_root
        self.class_num = class_num
        self.pretrained_model = pretrained_model
        self.isComplete = isComplete
        self.model_name = model_name

        # 动态加载模型模块
        model_module = importlib.import_module(f'model.{self.model_name}')
        self.net_class = getattr(model_module, 'self_net')

    def train(self, lr=1e-4, epochs=300, batch_size=4, print_freq=5, class_weights=None):
        """
        模型训练方法，执行模型的训练、验证和保存最佳模型，同时记录每一轮的loss和IOU数据。
        
        参数:
        lr : float, 初始学习率
        epochs : int, 训练的总轮数
        batch_size : int, 每个批次的样本数
        print_freq : int, 每隔多少次迭代输出一次损失信息
        class_weights : list, 类别权重
        """
        # 实例化模型
        net = self.net_class()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建以模型名称命名的日志目录
        log_dir = f'log/{self.model_name}/{timestamp}'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        pretrained_status = "None"
        if self.pretrained_model is not None and os.path.exists(self.pretrained_model):
            try:
                if self.isComplete:
                    net = torch.load(self.pretrained_model, map_location=self.device)
                else:
                    net.load_state_dict(torch.load(self.pretrained_model, map_location=self.device))
                pretrained_status = f"Loaded pretrained model from {self.pretrained_model}"
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                pretrained_status = "Error loading pretrained model"
        else:
            pretrained_status = "Pretrained model file not found"

        # 保存训练设置到文件
        settings_file = f'{log_dir}/training_settings_{timestamp}.txt'
        with open(settings_file, 'w') as f:
            f.write(f"Training started at: {timestamp}\n")
            f.write(f"Data root: {self.data_root}\n")
            f.write(f"Learning rate: {lr}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Pretrained model: {pretrained_status}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Class weights: {class_weights}\n")

        net = net.to(self.device)
        net.train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        train_set = Crack(self.data_root, 'train')
        val_set = Crack(self.data_root, 'test')
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size, shuffle=True,
            num_workers=64, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1, shuffle=True,
            num_workers=32, pin_memory=True)
        
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        miou = 0
        training_log = []

        for epoch in range(epochs):
            for param_group in optimizer.param_groups:
                lr = lr * (1 - (epoch / epochs) ** 2)
                param_group['lr'] = lr

            net.train()
            loss_epoch = []
            for i, (img, lab) in enumerate(train_loader):
                lab = lab.type(torch.LongTensor)
                img = img.type(torch.FloatTensor)
                img, lab = img.to(self.device), lab.to(self.device)
                pred = net(img)
                loss = criterion(pred, lab)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.data.cpu().numpy())

                if (i + 1) % print_freq == 0:
                    print('### iteration %5d of loss %5f epoch %5d' % 
                        (i + 1, float(np.array(loss_epoch).mean()), epoch+1))

            iou = [[], [], [], []]
            net.eval()
            with torch.no_grad():
                for img, lab in val_loader:
                    lab = lab.type(torch.LongTensor)
                    img = img.type(torch.FloatTensor)
                    img, lab = img.to(self.device), lab.to(self.device)
                    pred = net(img)
                    pred = pred.squeeze(0)
                    pred = torch.argmax(pred, dim=0, keepdim=True) 
                    pred = pred.cpu()
                    lab = lab.cpu()
                    iou_temp = compute_iou_with_matrix(pred, lab, self.class_num + 1)
                    for i in range(1, self.class_num + 1):
                        if iou_temp[i] >= 0:
                            iou[i].append(iou_temp[i])
                        
            iou_1 = np.mean(iou[1])
            iou_2 = np.mean(iou[2])
            iou_3 = np.mean(iou[3])
            miou_ = (iou_1 + iou_2 + iou_3) / 3

            print(f' epoch{epoch} / : iou1:{iou_1:.5f}, iou2:{iou_2:.5f}, iou3:{iou_3:.5f}, miou:{miou_:.5f}')
            
            training_log.append({
                'epoch': epoch,
                'loss': float(np.array(loss_epoch).mean()),
                'iou1': iou_1,
                'iou2': iou_2,
                'iou3': iou_3,
                'miou': miou_,
                'best_miou': miou
            })

            if (epoch + 1) % 5 == 0:
                df = pd.DataFrame(training_log)
                csv_path = f'{log_dir}/training_log_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                print(f"Training log updated and saved at epoch {epoch + 1}")
                self.plot_training_log(df, timestamp, log_dir)

            if miou_ > miou:
                miou = miou_
                file_name = f'./{log_dir}/self_best_model_{timestamp}.pth'
                torch.save(net, file_name)
                print(f'Saving the best model at the end of epoch {epoch}')

        df = pd.DataFrame(training_log)
        csv_path = f'{log_dir}/training_log_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Training log saved to '{csv_path}'")
        self.plot_training_log(df, timestamp, log_dir)

    def plot_training_log(self, df, timestamp, log_dir):
        """
        绘制训练日志中的损失值和IOU，并保存图像到指定的日志文件夹。
        
        参数:
        df : pandas.DataFrame, 训练日志数据
        timestamp : str, 时间戳
        log_dir : str, 日志保存目录
        """
        plt.figure(figsize=(10, 6))

        # 绘制Loss曲线
        plt.subplot(2, 1, 1)
        plt.plot(df['epoch'], df['loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.grid(True)
        plt.legend()

        # 绘制IOU曲线
        plt.subplot(2, 1, 2)
        plt.plot(df['epoch'], df['iou1'], label='IOU1')
        plt.plot(df['epoch'], df['iou2'], label='IOU2')
        plt.plot(df['epoch'], df['iou3'], label='IOU3')
        plt.plot(df['epoch'], df['miou'], label='Mean IOU')
        plt.plot(df['epoch'], df['best_miou'], label='Best mIOU')
        plt.xlabel('Epoch')
        plt.ylabel('IOU')
        plt.title('IOU over Epochs')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # 保存图像到模型日志目录
        plot_path = f'{log_dir}/training_log_plot_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Training log plot saved to '{plot_path}'")

# if __name__ == "__main__":
#     # 示例如何使用这个Train类进行训练
#     my_train = Train(model_name='zzy_model2')  # 这里的模型名称可以根据实际模型调整
#     class_weights = [8, 8, 12, 16]  # 根据类别分布设定权重
#     my_train.train(lr=5e-4, epochs=400, batch_size=4, print_freq=600, class_weights=class_weights)

