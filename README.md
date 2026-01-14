
# SkyGPT-ViTODE  
**Physics-Informed Neural ODE Framework for Probabilistic Ultra-Short-Term Solar Forecasting**

---

## 📌 Overview

**SkyGPT-ViTODE** is a physics-informed, probabilistic deep learning framework for **ultra-short-term solar photovoltaic (PV) power forecasting** using **ground-based sky images**.

The framework integrates:

- **Discrete visual representation learning** (Vector Quantized Variational Autoencoder, VQ-VAE)
- **Global spatiotemporal reasoning** (Vision Transformer, ViT)
- **Continuous-time dynamics modeling** (Neural Ordinary Differential Equations, Neural ODEs)
- **Physics-informed latent evolution** (PhyCell with conservation constraints)
- **Multimodal probabilistic forecasting** (Mixture Density Network, MDN)
- **Distribution-free uncertainty calibration** (Split Conformal Prediction)

SkyGPT-ViTODE produces **accurate, calibrated predictive distributions** with **formal coverage guarantees**, addressing key limitations of existing deterministic and probabilistic solar forecasting methods.

This repository contains the **full implementation used in the paper**:

> **“Physics-Informed Neural ODE Framework for Probabilistic Ultra-Short-Term Solar Forecasting”**  
> *Submitted to Applied Energy*

---

## ✨ Key Contributions

- Continuous-time modeling of cloud dynamics via **Neural ODEs**
- Physics-informed latent evolution using **moment-constrained PhyCells**
- Discrete sky-image tokenization with **VQ-VAE**
- Global spatial reasoning via **Vision Transformers**
- **Calibrated probabilistic forecasting** using MDN + conformal inference
- State-of-the-art performance on the **SKIPP'D benchmark**

---

## 📊 Main Results (SKIPP'D Dataset)

| Metric | Value |
|------|------|
| Forecast horizon | 15 minutes |
| nMAE | **7.89%** |
| MAE | **9.47 kW** |
| CRPS | **8.93 kW** |
| CRPS Skill | **68.9%** |
| Empirical Coverage (90%) | **90.4%** |
| Calibration Error | **0.018** |
| R² | **0.965** |

Statistical significance is confirmed via **Diebold–Mariano tests**  
(*p* < 10⁻¹⁰ against all baselines).

---

## 📂 Repository Structure

```text
SkyGPT-ViTODE/
├── Figures/                       # All figures used in the paper
│   ├── fig_architecture
│   ├── fig_ablation_alt
│   ├── fig_comprehensive_evaluation
│   ├── fig_multi_horizon_full
│   ├── fig_prediction_curves_7methods
│   ├── fig_research_gap_python_v2
│   ├── fig_stratification
│   └── phycell_integration
│
├── SkyGPT-ViTODE-config.py         # Experiment configuration and argument parsing
├── SkyGPT-ViTODE-data.py           # Dataset loading, preprocessing, and augmentation
├── SkyGPT-ViTODE-default.yaml      # Default experiment configuration (recommended)
├── SkyGPT-ViTODE-models.py         # Model definitions (VQ-VAE, ViT, ODE, PhyCell, MDN)
├── SkyGPT-ViTODE-train.py          # Training pipeline (PyTorch Lightning)
├── SkyGPT-ViTODE-evaluate.py       # Evaluation, metrics, calibration, and statistical tests
└── README.md                       # This file
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SkyGPT-ViTODE.git
cd SkyGPT-ViTODE
```

### 2. Create a virtual environment (recommended)
```bash
conda create -n skygpt-vitode python=3.10
conda activate skygpt-vitode
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies**
- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.0
- torchdiffeq
- NumPy, SciPy, scikit-learn
- matplotlib, seaborn
- PyYAML

---

## 📥 Dataset

### SKIPP'D Dataset
The experiments use the **SKIPP'D (SKy Images and Photovoltaic Power generation Dataset)**:

- 3 years of data (2017–2019)
- 84,803 daylight samples
- 379 kW PV system
- 1-minute sky images + synchronized PV power

**Download:**  
👉 https://github.com/yuhao-nie/stanford-solar-forecasting-dataset

Place the dataset path in `SkyGPT-ViTODE-default.yaml`.

---

## 🚀 Training

### Default training (recommended)
```bash
python SkyGPT-ViTODE-train.py   --config SkyGPT-ViTODE-default.yaml
```

### Key training features
- Distributed Data Parallel (DDP)
- Mixed precision (FP16)
- Gradient accumulation
- Early stopping on validation CRPS
- Automatic checkpointing

**Hardware used in the paper**
- 4 × NVIDIA RTX 3090 (24 GB)
- Training time: ~18.4 hours
- Total parameters: 47.3M

---

## 📈 Evaluation

Run full evaluation (deterministic + probabilistic):

```bash
python SkyGPT-ViTODE-evaluate.py   --config SkyGPT-ViTODE-default.yaml   --checkpoint path/to/best.ckpt
```

### Evaluation includes
- MAE, RMSE, nMAE, R²
- CRPS, Winkler Score
- Coverage at 80/90/95%
- Reliability diagrams
- PIT histograms
- Diebold–Mariano tests
- Bootstrap confidence intervals

---

## 📐 Conformal Prediction

SkyGPT-ViTODE applies **split conformal prediction** to MDN outputs:

- Distribution-free
- Finite-sample corrected
- Guaranteed marginal coverage under exchangeability

Prediction intervals adapt dynamically to forecast difficulty while maintaining calibration.

---

## 🔬 Reproducibility

- Fixed random seed (`seed = 42`)
- Deterministic CUDA operations
- Chronological train/val/test split
- All hyperparameters reported in the paper

---

---

## 🧠 Authors

- **Kombou Victor** – Conceptualization, Methodology, Software, Writing  
- **Kuiche Sop Brinda Leaticia** – Validation, Analysis  
- **Stephane Richard Befoum** – Software, Data curation  
- **Anto Leoba Jonathan** – Supervision, Methodology  

---

## 📜 License

This project is released under the **MIT License**.  
See `LICENSE` file for details.

---

## 🤝 Acknowledgements

- SKIPP'D dataset contributors  
- PyTorch and PyTorch Lightning communities  
- Open-source scientific computing ecosystem  

---

## 📬 Contact

For questions or collaboration:

📧 **kombouvictor@std.uestc.edu.cn**

---

**SkyGPT-ViTODE** provides a foundation for **operational, risk-aware solar forecasting** with calibrated uncertainty and physical consistency.
