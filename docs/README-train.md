```bash
python main.py --model_name 'zzy_model1' --data_root 'datasets_original' --lr 0.0005 --epochs 300 --batch_size 4 --class_weights '8,8,12,16'
```

```bash
python main.py --model_name 'zcrmodel_relu' --data_root 'datasets_original' --lr 0.0005 --epochs 300 --batch_size 4 --class_weights '8,8,12,16'
```

```bash
python main.py --model_name 'zcrmodel_leakyrelu' --data_root 'datasets_original' --lr 0.0005 --epochs 300 --batch_size 4 --class_weights '8,8,12,16'
```

```bash
python main.py --model_name 'fzx_model_copy' --data_root 'datasets_original_flip' --lr 0.0005 --epochs 400 --batch_size 4 --class_weights '8,16,12,16'
```

```bash
python main.py --model_name 'unet' --data_root 'datasets_original' --lr 0.0005 --epochs 400 --batch_size 4 --class_weights '8,8,12,16'
```

```bash
python main.py --model_name 'model' --data_root 'datasets_original' --lr 0.0005 --epochs 400 --batch_size 4 --class_weights '8,8,12,16'
```

python main.py --model_name 'fzx_model_copy' --data_root 'datasets_original' --lr 0.0005 --epochs 400 --batch_size 4 --class_weights '8,8,12,16'