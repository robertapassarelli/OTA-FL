# Federated Learning via Over-The-Air (OTA-FL)
Federated learning via-over-the air: algorithms and experimental validation

This repository contains the implementation and experimental framework developed for the Master’s Thesis:
> **"Federated Learning via over-the-air: Algorithms and Experimental Validation"**  
> *Roberta Passarelli, Sapienza University of Rome, 2025*  
> Supervisor: **Prof. Paolo Di Lorenzo**

---
## Overview
This project investigates **Federated Learning (FL)** and its **Over-the-Air (OTA-FL)** extension — a communication-efficient paradigm where model aggregation is performed directly through the wireless multiple-access channel.

The repository includes the complete implementation used for the **MNIST** and **EuroSAT** case studies described in the thesis, enabling reproducibility of the main results and plots.

---
## 📂 Repository Structure<pre>
```text
OTA-FL/
│
├── MNIST_FL.ipynb         # FL and OTA-FL experiments on MNIST dataset
├── EuroSAT_FL.ipynb       # FL and OTA-FL experiments on EuroSAT dataset
│
├── plots/                 # Example figures produced by the notebooks
│
├── requirements.txt       # Python environment dependencies
└── README.md
```
</pre>

> Only a **subset** of the plots (accuracy/loss) is provided to illustrate the experiment structure.  
> Full results can be reproduced by executing the notebooks or are available upon request.

---

## Methodology Summary

The experimental pipeline follows the structure presented in **Chapter 5** of the thesis:
1. **Data preparation** – download, normalization, and partition into 10 i.i.d. clients.
2. **System configuration** – set global hyperparameters (batch size, epochs, learning rate, number of communication rounds).
3. **Federated training** – perform training under:
   - *Ideal Federated Averaging (FedAvg)*;
   - *(Centralized) Over-the-Air Federated Learning (OTA-FL)* with noise and fading;
   - *Decentralized Over-the-Air Federated Learning* with noise and fading.
4. **Evaluation** – compute global model accuracy and loss after each communication round.
5. **Visualization** – plot comparative accuracy/loss curves for different settings.

---

## Datasets
### MNIST
- Handwritten digits dataset (28×28 grayscale images, 10 classes).
- Publicly available via [Kaggle](https://www.kaggle.com/datasets/scolianni/mnistasjpg) or directly from TensorFlow/Keras.
- Partitioned into 10 balanced clients for FL simulation.

### EuroSAT
- RGB satellite images (64×64, 10 land-use classes) from the Sentinel-2 dataset.
- Automatically downloaded via:
  wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
- Split 80/20 into train/test and distributed across 10 clients.

> Datasets are not included in this repository due to size and licence restrictions.
> They are automatically downloaded or imported when running the notebook.

---

## Implementation Details

- **Frameworks:** TensorFlow/Keras (MNIST) and PyTorch (EuroSAT)  
- **Optimizer:** SGD with momentum (0.9); SGD and Adam optional for EuroSAT experiments  
- **Loss:** Categorical cross-entropy / CrossEntropyLoss  
- **Communication rounds:** 100  
- **Clients:** 10 (all participating at each round)  
- **Batch size:** 32  
- **Local epochs:** 1 per round  

The OTA-FL implementation models channel effects as complex Gaussian noise and fading:
$$
h_k \sim \mathcal{CN}(1, \sigma_h^2), \quad n \sim \mathcal{CN}(0, \sigma_n^2)
$$

Different experimental configurations are simulated to analyze the behavior of OTA-FL under realistic wireless conditions.  
They are grouped according to **network topology** and **channel modeling**.

---
### Centralized OTA-FL

Model aggregation is performed by a single central server (base station) that receives the superimposed updates from all clients through the wireless multiple-access channel.  
The following cases are considered:
1. **Ideal FL (FedAvg)** – classical digital aggregation without any channel distortion.  
2. **OTA-FL (no fading)** – the channel gain is unitary ($h_k = 1$ ∀k); only additive Gaussian noise is present.  
3. **OTA-FL (fading + noise)** – each client experiences random complex fading ($h_k \sim \mathcal{CN}(1, \sigma_h^2)$) and additive noise ($n \sim \mathcal{CN}(0, \sigma_n^2)$), representing a realistic wireless environment.  
4. **Ideal-training tested OTA** – the model is trained in ideal FL conditions and then evaluated under OTA transmission impairments (noise and fading). This setup isolates the *pure effect of wireless aggregation* on model performance.

---

### Decentralized OTA-FL
In this configuration, there is **no central server**.  
Each client communicates directly with its neighbors according to a defined graph topology (e.g., Erdős–Rényi or structured graphs).  
Model aggregation occurs *over the air* within local neighborhoods via superposition of transmitted updates.

1. **Decentralized OTA-FL (no fading)** – peer-to-peer aggregation with additive noise only.  
2. **Decentralized OTA-FL (fading + noise)** – peer-to-peer aggregation under both fading and noise, modeling realistic distributed communication.

---
This organization allows comparing:
- the **impact of the communication topology** (centralized vs. decentralized), and  
- the **effect of channel distortions** on convergence and accuracy under both ideal and over-the-air aggregation.
---

## How to Run

### Option 1 — Google Colab
Open directly in Colab:
- [MNIST_FL.ipynb](https://colab.research.google.com/github/robertapassarelli/OTA-FL/blob/main/MNIST_FL.ipynb)
- [EuroSAT_FL.ipynb](https://colab.research.google.com/github/robertapassarelli/OTA-FL/blob/main/EuroSAT_FL.ipynb)

The notebooks automatically handle dataset import, training, and plotting.

### Option 2 — Local environment
```bash
git clone https://github.com/USERNAME/Federated-Learning-OTA.git
cd Federated-Learning-OTA
pip install -r requirements.txt
jupyter notebook
```
---

## Reproducibility

All experiments are designed to ensure **full reproducibility** of the results presented in the thesis.  
Each Jupyter notebook (`MNIST_FL.ipynb`, `EuroSAT_FL.ipynb`) contains the **complete training pipeline**, from data preprocessing to model evaluation, following the experimental setup described in **Chapter 5 – Implementation and Experimental Setup** of the thesis.

- Random seeds are fixed for deterministic behavior where possible.  
- All hyperparameters (learning rate, batch size, local epochs, number of communication rounds) are kept consistent across experiments.  
- The dataset partitioning, channel modeling, and training procedures are implemented identically for FedAvg and OTA-FL configurations.  

> The repository is self-contained: running the notebooks reproduces the main accuracy and loss curves discussed in **Chapter 6 – Numerical Results and Discussion**.

---

## Citation
If you use this repository, please cite:
@mastersthesis{passarelli2025,
  title     = {Federated Learning over Wireless Networks},
  author    = {Roberta Passarelli},
  school    = {Sapienza University of Rome},
  year      = {2025},
  advisor   = {Prof. Paolo Di Lorenzo}
}
You may also include a reference to this GitHub repository:
https://github.com/USERNAME/Federated-Learning-OTA

## Contact

Roberta Passarelli
Master’s Degree in Data Science – Sapienza University of Rome
📧 passarelli.1466794@studenti.uniroma1.it
🔗 https://github.com/robertapassarelli 

---

## License 
This project is released under the MIT License.
You are free to use, modify, and distribute the code with proper attribution.


*“All implementation code and experimental configurations are publicly available to ensure full reproducibility of the results presented in the thesis.”*
