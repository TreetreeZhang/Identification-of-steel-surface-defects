import argparse
from scripts.train import Train
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument('--model_name', type=str, default='zzy_model2', help='Name of the model to be used for training.')
    parser.add_argument('--data_root', type=str, default='datasets_origin_enhance', help='Root directory of the dataset.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--class_num', type=int, default=3, help='Number of classes for segmentation or classification task.')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to the pretrained model. Set to None if no pretrained model is used.')
    parser.add_argument('--isComplete', type=bool, default=True, help='Whether to load the complete pretrained model or only the state_dict.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--print_freq', type=int, default=600, help='How often to print loss information during training.')
    parser.add_argument('--class_weights', type=str, default='8,8,12,16', help='Class weights, separated by commas.')

    args = parser.parse_args()

    # 将 class_weights 从字符串转换为浮点数列表
    class_weights = [float(weight) for weight in args.class_weights.split(',')]

    # 实例化 Train 类
    trainer = Train(model_name=args.model_name, 
                    data_root=args.data_root, 
                    device=args.device, 
                    class_num=args.class_num, 
                    pretrained_model=args.pretrained_model, 
                    isComplete=args.isComplete)

    # 调用训练方法
    trainer.train(lr=args.lr, 
                  epochs=args.epochs, 
                  batch_size=args.batch_size, 
                  print_freq=args.print_freq, 
                  class_weights=class_weights)

if __name__ == "__main__":
    main()
