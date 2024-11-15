
# Identification-of-steel-surface-defects

This repository provides methods using **Unet and two kinds of its variants** for **defect recognition and semantic segmentation on steel surfaces**. The main contributors to this work are [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📖 Introduction

### What is Steel surface defect recognition?  
Steel surface defect recognition focuses on detecting and classifying surface imperfections to ensure quality control. It utilizes techniques such as image processing for defect segmentation and deep learning models, like Convolutional Neural Networks (CNNs), for accurate classification of defects such as cracks, scratches, and pits. This technology is crucial in automating quality inspection in steel manufacturing.

This repository uses **Unet and its variants** to perform defect recognition and semantic segmentation on a set of steel surface images.

### Source of the dataset

The data in this repository comes from a dataset created by Professor Song Kechen's team at **Northeastern University**, which includes three types of defects: **patches**, **inclusion**, and **scratches**.

### 技术路线
1. **模型变迁**:我们的Baseline是Unet，在此基础上又得到了两个不同的Unet变种模型。
2. **训练方法**：采用LMS损失函数，Admaw优化器进行训练。
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

### 4. Running the Algorithms

To run a selected model or specify other hyperparameters., execute the following command:

```bash
python main.py --model_name 'Unet' --data_root 'datasets' --lr 0.0005 --epochs 300 --batch_size 4 --class_weights '8,8,12,16'
```

Replace `Unet` with `model1` or `model2` depending on the model_name you want to test. 
Replace `datasets` with `datasets1` or `datasets2` depending on the dataset you want to test.
You can also modify **hyperparameters** within the algorithm scripts inside the `methods/` directory.

---

## 📊 Results



---

## 📂 Directory Structure

```

```

---

## 🔧 Tuning and Customization


---

## 📬 Contact

For questions or feedback, feel free to open an issue or contact the repository owner via GitHub: [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📚Reference

Wenli Zhao,  Kechen Song,  Yanyan Wang, Shubo Liang, Yunhui Yan. FaNet: Feature-aware Network for Few Shot Classification of Strip Steel Surface Defects [J].  
