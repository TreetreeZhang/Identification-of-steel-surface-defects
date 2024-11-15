
# Identification-of-steel-surface-defects

This repository provides methods using **Unet and two kinds of its variants** for **defect recognition and semantic segmentation on steel surfaces**. The main contributors to this work are [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📖 Introduction

### What is Steel surface defect recognition?  
Steel surface defect recognition focuses on detecting and classifying surface imperfections to ensure quality control. It utilizes techniques such as image processing for defect segmentation and deep learning models, like Convolutional Neural Networks (CNNs), for accurate classification of defects such as cracks, scratches, and pits. This technology is crucial in automating quality inspection in steel manufacturing.

This repository uses **Unet and its variants** to perform defect recognition and semantic segmentation on a set of steel surface images.

### Source of the dataset

The data in this repository comes from a dataset created by Professor Song Kechen's team at **Northeastern University**, which includes three types of defects: **patches**, **inclusion**, and **scratches**.

At first, we used the classic Unet for prediction. In the subsequent process, in order to reduce the number of model parameters, we developed two Unet variant models.
---

## ⚙️ Algorithms Implemented
1. **Genetic Algorithm (GA)**: Mimics evolution through selection, crossover, and mutation to explore the solution space.
2. **Simulated Annealing (SA)**: Models the cooling of metals, gradually refining solutions to escape local optima.
3. **Particle Swarm Optimization (PSO)**: Simulates the social behavior of birds to explore the solution space collectively.

These algorithms are ideal for **avoiding local minima** and **exploring large solution spaces** efficiently.

---

## 🚀 Getting Started
### 1. Clone to a local computer

You can download the zip file or run the following command:

```
git clone https://github.com/TreetreeZhang/Identification-of-steel-surface-defects.git
cd Identification-of-steel-surface-defects
```


### 2. Installation

Ensure Python is installed, then set up the environment:

```bash
pip install -e .
pip install -r requirements.txt
```

---

### 3. Data Preparation
You can use the provided example datasets (e.g., `mk01.txt`) or prepare your own. Custom data should follow this format:

```
datasets/
│
├── test/
.   │
.   ├──image/               #Store the origin images.
.   └──label/               #Store the mask images.
└── train/
    │
    ├──image/               #Store the origin images.
    └──label/               #Store the mask images.



```



---

### 4. Training the Model

To train a selected model with specified hyperparameters, execute the following command:

```bash
python main.py --model_name 'Unet' --data_root 'datasets' --lr 0.0005 --epochs 300 --batch_size 4 --class_weights '8,8,12,16' --pretrained_model 'model.pth'
```

Ensure that you replace:
- `Unet` with either `model1` or `model2` depending on the model you wish to test.
- `datasets` with either `datasets1` or `datasets2` depending on the dataset you intend to use for training.

Additionally, you may modify **hyperparameters** within the algorithm scripts located in the `methods/` directory.

#### Parameters:
- `lr` (Learning Rate): Defines the learning rate for model training.
- `epochs`: Specifies the number of training epochs.
- `batch_size`: Defines the batch size for training.
- `class_weights`: Specifies the class weights for the loss function, where:
  - `class_weights[0]` corresponds to the background class weight,
  - `class_weights[1]` corresponds to the **patches** class weight,
  - `class_weights[2]` corresponds to the **inclusion** class weight, and
  - `class_weights[3]` corresponds to the **scratches** class weight.
- `pretrained_model`: Path to the pretrained model weights (e.g., `model.pth`).

---

### 5. Running the Prediction Algorithm

To make predictions using a trained model, execute the following command:

```bash
python predict.py --model_path 'model.pth' --data_root 'datasets' --num_classes 4
```

Make sure to replace:
- `model.pth` with the path to the model you wish to use for prediction (e.g., `model1.pth` or `model2.pth`).
- `datasets` with either `datasets1` or `datasets2` depending on the dataset you intend to test.

You may also modify **hyperparameters** within the algorithm scripts located in the `methods/` directory.
---

## 📊 Results



---

## 📬 Contact

For questions or feedback, feel free to open an issue or contact the repository owner via GitHub: [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📚Reference

Wenli Zhao,  Kechen Song,  Yanyan Wang, Shubo Liang, Yunhui Yan. FaNet: Feature-aware Network for Few Shot Classification of Strip Steel Surface Defects [J].  
