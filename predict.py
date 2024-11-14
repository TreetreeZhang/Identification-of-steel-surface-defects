import os
import argparse
import torch
from scripts.predict import Predict  # Import the Predict class

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Use the trained model for prediction and evaluation")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--data_root', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of classes, default is 4")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help="Device to use, 'cpu' or 'cuda'. Default is 'cuda'")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure the dataset path is an absolute path (optional)
    print("Dataset path:", os.path.abspath(args.data_root))

    # Set device for computation
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the Predict class and perform evaluation
    predictor = Predict(model_path=args.model_path, data_root=args.data_root, device=device)
    predictor.evaluate(num_classes=args.num_classes)

if __name__ == "__main__":
    main()
