
# Identification-of-steel-surface-defects

This repository provides methods using **Unet and two kinds of its variants** for **defect recognition and semantic segmentation on steel surfaces**. The main contributors to this work are [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📖 Introduction

### What is Steel surface defect recognition?  
Steel surface defect recognition focuses on detecting and classifying surface imperfections to ensure quality control. It utilizes techniques such as image processing for defect segmentation and deep learning models, like Convolutional Neural Networks (CNNs), for accurate classification of defects such as cracks, scratches, and pits. This technology is crucial in automating quality inspection in steel manufacturing.

This repository uses **Unet and its variants** to perform defect recognition and semantic segmentation on a set of steel surface images.

### Source of the dataset

The data in this repository comes from a dataset created by Professor Song Kechen's team at **Northeastern University**, which includes three types of defects: **patches**, **inclusion**, and **scratches**.

---

## ⚙️ Algorithms Implemented
1. **Genetic Algorithm (GA)**: Mimics evolution through selection, crossover, and mutation to explore the solution space.
2. **Simulated Annealing (SA)**: Models the cooling of metals, gradually refining solutions to escape local optima.
3. **Particle Swarm Optimization (PSO)**: Simulates the social behavior of birds to explore the solution space collectively.

These algorithms are ideal for **avoiding local minima** and **exploring large solution spaces** efficiently.

---

## 🚀 Getting Started

### 1. Installation
Ensure Python is installed, then set up the environment:

```bash
pip install -e .
pip install -r requirements.txt
```

---

### 2. Data Preparation
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

### 3. Running the Algorithms

To run a selected model or specify other hyperparameters., execute the following command:

```bash
python main.py --solver GA --datapath ./data/mk01.json
```

Replace `GA` with `SA` or `PSO` depending on the algorithm you want to test. You can also modify **hyperparameters** within the algorithm scripts inside the `methods/` directory.

---

## 📊 Results

Results will be saved to the **`results/`** folder. Each result includes the following:
- **Makespan** (total completion time)
- **Machine utilization**
- **Gantt chart visualization** (optional: for visualizing schedules)

---

## 📂 Directory Structure

```
Metaheuristic-Algorithms-For-JSP-and-FJSP-Problems/
│
├── data/                # Sample datasets
├── methods/             # Algorithm implementations
├── results/             # Output files
├── utils/               # Helper functions
├── main.py              # Entry point for running algorithms
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🔧 Tuning and Customization

To modify the behavior of each algorithm:
- **GA**: Adjust mutation and crossover rates.
- **SA**: Modify the cooling schedule or initial temperature.
- **PSO**: Tune the inertia weight or cognitive/social parameters.

---

## 📬 Contact

For questions or feedback, feel free to open an issue or contact the repository owner via GitHub: [Zhang Chaorui](https://github.com/TreetreeZhang), [Zhang Zhenyu](https://github.com/LittleRookie1115), and [Fang Zixiang](https://github.com/TtLuckyyy).

---

## 📚Reference

Wenli Zhao,  Kechen Song,  Yanyan Wang, Shubo Liang, Yunhui Yan. FaNet: Feature-aware Network for Few Shot Classification of Strip Steel Surface Defects [J].  