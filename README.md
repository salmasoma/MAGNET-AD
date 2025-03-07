# 🧠 MAGNET-AD: Multitask Spatiotemporal GNN for Alzheimer's Prediction

## 📌 Overview

MAGNET-AD is a novel multitask spatiotemporal graph neural network (STGNN) designed to predict both the Preclinical Alzheimer's Cognitive Composite (PACC) score and time to AD conversion. It achieves state-of-the-art performance by integrating multimodal data and capturing the complex interplay of biological, structural, and temporal factors in preclinical Alzheimer's Disease.

This repository contains the official inference code for the MAGNET-AD (Multitask Spatiotemporal GNN for Interpretable Prediction of PACC and Conversion Time in Preclinical Alzheimer) framework.


## 🏗️ Architecture

![Model Architecture](Figures/MAGNET_AD_Arch.png)

The framework consists of four key components:

1. 🧩  **Hybrid Data Fusion** : Integrates dynamic neuroimaging patterns with time-invariant genetic markers through weighted edges
2. ⏱️  **Dual Attention Mechanisms** : Employs spatial attention for relationships between brain structures and genetic factors, and temporal attention for structural changes across visits
3. 📊  **Multi-Task Learning** : Simultaneously predicts PACC scores and AD conversion time through specialized prediction heads
4. 📈  **Temporal Importance Weighting** : Adaptively learns critical time points in disease progression using an innovative loss function

## 🛠️ Installation

Ensure you have the required dependencies installed:

```python
# Create Environment
conda create -n magnetad python=3.9
# Activate Environment
conda activate magnetad
# Clone Repo
git clone https://github.com/salmasoma/MAGNET-AD/
cd MAGNET-AD
# Install requirements
pip install -r requirements.txt
```

Download model weights: [Weights](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/salma_hassan_mbzuai_ac_ae/EQJEbpgb8yBLnQtp3tLxaFQBgbW4NJ8Uymm7lDF7Q7EpmA?e=NPyjOm)

## 🚀 Usage

### Generate brain graphs separately:

```python
python generate_graphs.py \
  --embeddings_dir "/path/to/structure_embeddings" \
  --correlation_path "/path/to/bold_correlations.csv" \
  --clinical_data "/path/to/clinical_data.csv" \
  --gene_gene_path "/path/to/gene_gene.csv" \
  --gene_structure_path "/path/to/gene_structure.csv" \
  --gene_embeddings_dir "/path/to/gene_embeddings" \
  --output_dir "./brain_graphs" \
  --bold_thresholds 50 70 90 100\
  --create_splits
```

### Run inference on pre-generated graphs:

```python
python simple_inference.py \
  --model_path "/path/to/trained_model.pt" \
  --data_dir "./brain_graphs" \
  --csv_file "/path/to/clinical_data.csv" \
  --output_file "./results/predictions.pkl" \
  --bold_threshold 100
```

### Generate graphs and run inference on patient data:

```python
python complete_inference.py \
  --model_path "models/magnetad_model.pt" \
  --embeddings_dir "/path/to/structure_embeddings" \
  --correlation_path "/path/to/bold_correlations.csv" \
  --clinical_data "/path/to/clinical_data.csv" \
  --gene_gene_path "/path/to/gene_gene.csv" \
  --gene_structure_path "/path/to/gene_structure.csv" \
  --gene_embeddings_dir "/path/to/gene_embeddings" \
  --output_dir "./results" \
  --verbose
```

### Command line arguments for inference:

* `--model_path`: Path to the saved model (required)
* `--data_dir`: Directory containing brain graph data (required)
* `--csv_file`: Path to patient clinical CSV file (required)
* `--output_file`: Path to save prediction results (default: "./predictions.pkl")
* `--patient_ids`: Comma-separated list of patient IDs (optional)
* `--bold_threshold`: BOLD correlation threshold value (default: 50)

## 📊 Performance

MAGNET-AD achieves state-of-the-art performance in preclinical AD prediction:

* 🔹 Concordance Index: 0.858 for conversion time prediction
* 🔹 Mean Square Error: 1.983 for PACC prediction
* 🔹 Superior performance across various timepoints, even with limited visit data
* 🔹 Outperforms existing approaches (LSTM, TCN, Transformer, and GCN STGNN)

## 📂 Required Data Files

The model requires the following data files:

* 📋  **Clinical Data** : Patient information including survival times and PACC scores
* 🧠  **Brain Structure Embeddings** : 512-dimensional embeddings for each brain structure
* 🔄  **BOLD Correlations** : Functional connectivity between brain regions
* 🧬  **Gene-Gene Interactions** : Information about gene co-expression patterns
* 🔗  **Gene-Structure Connections** : Relationships between genes and brain structures
* 🧪  **Gene Embeddings** : 768-dimensional gene embeddings
* 📊 **Radiomics Data** (optional): Region-specific radiomics features for temporal edges

## 🔍 Interpretability

MAGNET-AD provides clinically relevant insights through:

* 🧠 Attention-based visualization of critical brain regions across time
* 🧬 Identification of key gene-to-structure connections in disease progression
* 📈 Learned temporal importance weights that highlight critical time points
* 🔄 Alignment with established Braak staging of AD pathology

## 🔄 Model Variants

Experiment with different model configurations:

* MRI-only: Using just neuroimaging data
* MRI+Genetic: Integrating both modalities
* With/without temporal weights: Control the importance of time-dependent features
* With/without radiomics features: Include radiomics features for edge weights
