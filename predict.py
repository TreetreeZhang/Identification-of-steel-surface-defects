import argparse
from scripts.predict import Predict 

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--data_root', type=str, default='./Dataset', help='Root directory of the dataset.')
    parser.add_argument('--device', type=str, default=None, help='Device to use for prediction (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for the prediction.')

    args = parser.parse_args()

    # 实例化 Predict 类并加载模型
    predictor = Predict(model_path=args.model_path, 
                        data_root=args.data_root, 
                        device=args.device)

    # 进行评估
    predictor.evaluate(num_classes=args.num_classes)

if __name__ == "__main__":
    main()
