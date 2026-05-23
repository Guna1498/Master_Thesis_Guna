# Generative AI for Data Analysis of Physical Structures from Aerial Images and Tabular Data

> **Master's Thesis** | Gunasekhar Jenni | Paderborn University, 2025  
> Department of Computer Science · Heinz Nixdorf Institute (HNI) · Data Science Group (DICE)

---

## Overview

This repository contains the code, notebooks, and supporting materials for a master's thesis that investigates the use of **open-source Vision Language Models (VLMs)** specifically `Qwen2.5-VL-7B-Instruct` to analyze physical building structures from aerial imagery combined with structured tabular geospatial data.

The core use case is **attic habitability classification** of residential buildings in Kassel, Germany, using data from the [UrbanTwin](https://www.iee.fraunhofer.de/) platform at Fraunhofer IEE. The system evaluates three inference modes:

| Mode | Description |
|------|-------------|
| 🖼️ Image-only | Zero-shot / few-shot VLM inference on aerial roof images |
| 📋 Tabular-only | Rule-based prompts over structured building metadata |
| 🔀 Multimodal | Combined tabular + image inference via prompt fusion |

---

## Abstract

This thesis explores the use of Generative AI, particularly Vision Language Models (VLMs), to analyze physical structures from aerial imagery in combination with structured tabular data. The study evaluates how well open-source VLMs can interpret architectural features such as skylights, dormers, and roof types through zero-shot prompting and multimodal reasoning tasks. These visual cues are then linked to corresponding attributes in structured data, enabling insights into use cases like assessing attic livability.

The findings reveal that while current models show promise in identifying fine-grained features, there are notable limitations in consistency and localization accuracy. This work highlights the potential of multimodal AI to automate and scale geospatial analysis and provides a foundation for future systems that integrate image and tabular data more effectively.

---

## Research Questions

1. **How effectively can VLMs extract and interpret features from aerial imagery?**
   - How accurately can open-source VLMs identify dormers and skylights in aerial imagery of gable and hipped residential roofs?
   - How do different prompting techniques and system-role formulations affect VLM performance?
   - What are the key limitations of current VLMs in this domain?

2. **How can visual data be linked with structured tabular data for combined inference?**
   - What methods can associate image-based features with tabular dataset rows?
   - Does combined inference outperform single-modality analysis?

3. **How can system outputs be validated against ground truth for reliability?**
   - What metrics best assess image-only, table-only, and combined inference?
   - How does performance vary across roof types (gable vs. hipped)?

---

## Methodology

### 1. Model Selection

The primary model selected after systematic leaderboard benchmarking and hands-on evaluation is:

**`Qwen2.5-VL-7B-Instruct`** (via Hugging Face)

Selection criteria included: open-source availability, multimodal reasoning capability, compute feasibility, and performance on public VLM benchmarks. Other candidates reviewed include LLaVA, InternVL2, Phi-3.5-Vision, and MiniGPT-4.

### 2. Dataset

**Aerial Imagery**
- Source: UrbanTwin platform (Fraunhofer IEE), Kassel, Germany
- 745 annotated images (375 gable-roof, 370 hipped-roof buildings)
- Annotation tool: [MakeSense.ai](https://www.makesense.ai/) with cross-verification via Google Earth
- Labels: dormer bounding boxes, skylight bounding boxes, roof type

**Tabular Metadata** (PostgreSQL, sourced from [BKG](https://www.bkg.bund.de/))

| Attribute | Description |
|-----------|-------------|
| `Gid` | Unique building ID (matches image filename) |
| `Roof_type` | 3100 = Gabled, 3200 = Hipped |
| `Slope` | Roof inclination angle (degrees) |
| `Roof_height_max` | Maximum roof height (meters) |
| `Roof_area` | Total surface area of roof (m²) |
| `Roof_volume` | Volume under the roof (m³) |
| `Floors` | Number of storeys |
| `Nutzfläche` | Usable horizontal floor area (m²) |

### 3. Ground Truth Definition

Two attic-habitability cases were defined:

**Case 1 Actual Attic Usage (strict AND logic):**
```
(slope ≥ 30° AND roof_height_max ≥ 4 m) OR (dormer = 1)
```

**Case 2 Conversion Potential (relaxed OR logic):**
```
(slope ≥ 30° OR roof_height_max ≥ 4 m) OR (dormer = 1)
```

Thresholds are grounded in building standards: a habitable attic requires ≥ 2 m ceiling height over half the net floor area, informing the 4 m roof height threshold.

### 4. Prompt Engineering

Three zero-shot prompt variants (P1–P3) were tested, ranging from generic to architecturally detailed. Two system roles were evaluated: `"Helpful Assistant"` vs. `"Expert"`. Few-shot prompting was applied to the best-performing zero-shot configuration.

---

## Key Findings

### Image-Only Inference (Experiment 1)

Performance across 6 prompt/persona combinations on Dataset 2 (sloped roofs only):

| Case | System Role | Prompt | Accuracy | Precision | Recall | F1 |
|------|-------------|--------|----------|-----------|--------|----|
| 1 | Helpful Assistant | P1 | 0.18 | 0.00 | 0.00 | 0.00 |
| 2 | Helpful Assistant | P2 | 0.20 | 0.80 | 0.02 | 0.05 |
| 3 | Helpful Assistant | P3 | 0.28 | 0.75 | 0.18 | 0.30 |
| 4 | Expert | P1 | 0.18 | 0.00 | 0.00 | 0.00 |
| 5 | Expert | P2 | 0.42 | 0.81 | 0.37 | 0.51 |
| **6** | **Expert** | **P3** | **0.48** | **0.79** | **0.49** | **0.61** |

**Few-shot (P3 + Expert):** Accuracy 0.816, F1 0.899, Recall 1.00 but at the cost of zero true negatives (over-prediction bias).

### Tabular-Only Inference (Experiment 2)

| Roof Subset | Logic | Precision | Recall | F1 |
|-------------|-------|-----------|--------|----|
| Gable | AND | 1.00 | 0.48 | 0.65 |
| Gable | OR | 1.00 | 0.77 | 0.87 |
| Hipped | AND | 0.99 | 0.45 | 0.62 |
| Hipped | OR | 0.99 | 0.63 | 0.77 |
| All Roofs | AND | | 0.47 | 0.64 |
| All Roofs | OR | | 0.71 | 0.83 |

OR-logic prompts consistently outperform AND-logic in recall and F1.

### Multimodal Inference (Tabular + Image)

| Case | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| AND | 0.69 | 0.69 | 0.67 | 0.68 |
| OR | 0.68 | 0.79 | 0.65 | 0.71 |

Combined inference offers marginal gains over tabular-only, with ambiguity arising when visual and tabular signals conflict (e.g., a dormer is visible but tabular thresholds aren't met).

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` (Hugging Face)
- `torch`
- `Pillow`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `psycopg2` (PostgreSQL connection)
- `geopandas`, `shapely` (geospatial operations)

### Running Inference

**Image-only inference:**
```bash
jupyter notebook notebooks/03_image_inference.ipynb
```

**Tabular-only inference:**
```bash
jupyter notebook notebooks/04_tabular_inference.ipynb
```

**Multimodal inference:**
```bash
jupyter notebook notebooks/05_multimodal_inference.ipynb
```

### Model Download

The model is loaded automatically from Hugging Face:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

---

## Thesis

The full thesis document is available in the `thesis/` directory:

📄 **[Generative AI for Data Analysis of Physical Structures from Aerial Images and Tabular Data](./thesis/)**  
Gunasekhar Jenni | Paderborn University, July 2025  
Reviewers: Dr. Stefan Heindorf, Dr. Michael Röder  
Supervisor: Daniel Horst (Fraunhofer IEE)

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{jenni2025genai,
  author    = {Gunasekhar Jenni},
  title     = {Generative AI for Data Analysis of Physical Structures from Aerial Images and Tabular Data},
  school    = {Paderborn University},
  year      = {2025},
  month     = {July},
  address   = {Paderborn, Germany},
  note      = {Data Science Group (DICE), Heinz Nixdorf Institute (HNI)}
}
```

---

## Acknowledgements

This research was conducted in collaboration with the **UrbanTwin** project at **Fraunhofer IEE**. Special thanks to Dr. Stefan Heindorf (Paderborn University), Daniel Horst (Fraunhofer IEE), and Dr. Michael Röder (DICE) for their guidance and review. The open-source community particularly Hugging Face provided the tools and libraries that made this work possible.

---

## License

This project is released for academic and research purposes. Please refer to individual data and model licenses (BKG data, Qwen2.5-VL) before using in production contexts.
