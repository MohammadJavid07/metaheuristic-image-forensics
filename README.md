# metaheuristic-image-forensics
Separating semantically edited image regions from generic AI generation traces using metaheuristic region selection (BGA/BPSO) over forensic trace maps and semantic indicators on the TGIF2 dataset.

# Metaheuristic Image Forensics: Separating Semantically Edited Regions via Region Selection

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![DEAP Framework](https://img.shields.io/badge/DEAP-BGA-orange.svg)](https://github.com/DEAP/deap)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

This repository contains the implementation of the course project for **Meta-Heuristic Methods** at Atatürk University, Department of Software Engineering. 

* **Course Instructor:** Dr. Öğr. Üyesi Şeyma Yücel Altay
* **Objective:** Formulate image tampering localization as a binary subset selection problem, utilizing a **Binary Genetic Algorithm (BGA)** guided by dual-signal forensic/semantic features and a parsimony penalty to isolate tampered regions without over-selection.

---

## 🏗️ System Architecture & Methodology

The pipeline processes tampered images through four modular stages:

1. **Superpixel Candidate Proposals (`src/region_proposals.py`)**: 
   * Utilizes **SLIC (Simple Linear Iterative Clustering)** to segment the input image into perceptually meaningful superpixel regions.
2. **Dual-Signal Feature Extraction (`src/feature_extractor.py`)**: 
   * Computes forensic high-frequency gradient statistics (Sobel edge energy) and semantic variance signals for each candidate region.
3. **Metaheuristic Optimization (`src/optimizers.py`)**: 
   * Implements a **DEAP-based Binary Genetic Algorithm (BGA)** optimizing a mean-based joint fitness function:
     $$\mathcal{F} = \alpha \bar{f} + \beta \bar{s} - \lambda \left(\frac{k}{N}\right)$$
     * Where $\bar{f}$ and $\bar{s}$ represent mean forensic and semantic scores, $k/N$ is the parsimony ratio, and $\lambda$ controls the penalty against over-selection.
4. **Quantitative Evaluation & Logging (`src/evaluate.py`, `src/main.py`)**: 
   * Reconstructs prediction masks and evaluates performance using **IoU (Intersection over Union)**, **F1-Score**, **Precision**, and **Recall**, with automated CSV logging and visualization exporting.

---

## 📁 Repository Structure

```text
metaheuristic-image-forensics/
│
├── data/
│   └── tgif2_sample/
│       ├── images/        # Input tampered images (synthetic/dataset samples)
│       └── masks/         # Ground-truth binary masks
│
├── outputs/
│   ├── experiment_log.csv # Automated metrics log across lambda parameter sweeps
│   ├── prediction_result.png
│   └── lambda_performance_curve.png
│
├── src/
│   ├── load_data.py       # Dataset loader & synthetic textured sample generator
│   ├── region_proposals.py# SLIC superpixel segmentation module
│   ├── feature_extractor.py# Dual-signal forensic & semantic feature extraction
│   ├── optimizers.py      # DEAP Binary Genetic Algorithm & fitness evaluation
│   ├── evaluate.py        # Mask reconstruction and metric calculation
│   ├── main.py            # End-to-end pipeline orchestrator & experiment sweeper
│   └── plot_results.py    # Dependency-free performance curve plotter
│
├── requirements.txt       # Project dependencies
└── README.md
