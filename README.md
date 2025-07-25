# Master Thesis: Generative AI for Data Analysis of Physical Structures from Aerial Images and Tabular Data

This repository contains all experimental notebooks developed as part of the master's thesis titled  
**"Generative AI for Data Analysis of Physical Structures from Aerial Images and Tabular Data"**.  


### Descriptions

- **Image Inference Experiment/**  
  Contains zero-shot and few-shot evaluations of VLMs for extracting visual features (e.g., skylights, dormers, roof type) directly from aerial images.

- **Tabular And Image Experiment/**  
  Evaluates attic usability based only on structured building metadata. Includes experiments specific to gable roofs, hipped roofs, and the combined dataset.

- **Tabular inference Experiment/**  
  Performs multimodal inference combining tabular data with aerial imagery to assess whether image input enhances predictions.

## Experimental Setup

- **Hardware**: All experiments were performed on a high-performance GPU node equipped with an **NVIDIA A100** (40GB).
- **Dataset**: A proprietary building-level dataset provided by **Fraunhofer IEE** was used. The dataset includes high-resolution aerial imagery and rich tabular metadata (e.g., roof type, height, slope, attic attributes). Data preparation involved intensive cleaning, standardization, and image-to-record alignment.
- **Note**: Due to confidentiality agreements, the dataset is not publicly available and cannot be redistributed.

## Requirements

- Python 3.8+
- Jupyter Notebook
- Open-source VLMs (Qwen-VL)
- Dependencies: `transformers`, `torch`, `matplotlib`, `pandas`, `numpy`




