# IndAgri-VLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **IndAgri-VLM: A Telugu-Grounded Multimodal Vision-Language Model for
> South Indian Commercial Crop Disease Diagnosis with Seasonal Context Conditioning**
>
> Dr. P.V.V. Kishore — KL University (KLEF), Department of Generative AI and Machine Learning

---

## Overview

IndAgri-VLM is a domain-adapted vision-language model fine-tuned on South Indian
commercial crops (Guntur chilli, Telangana cotton, coconut, groundnut, turmeric,
Bengal gram). It is the first VLM to:

- Cover India-specific commercial crops absent from AgroGPT, Agri-LLaVA, and WisWheat
- Generate disease diagnosis and treatment prescriptions in **Telugu**
- Condition diagnosis on **30-day seasonal weather context** (NASA POWER)

---

## Repository Structure
---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/pvvkishore/IndAgri-VLM.git
cd IndAgri-VLM

# 2. Create environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (see docs/DATA.md)
python scripts/download_data.py

# 5. Train
python scripts/train.py --config configs/qlora_qwen25vl_3b.yaml

# 6. Evaluate
python scripts/evaluate.py --config configs/eval.yaml
```

---

## Datasets

| Dataset     | Crops                        | Images  | Source              |
|-------------|------------------------------|---------|---------------------|
| PlantVillage| 14 crops, 26 diseases        | 54,309  | Hughes & Salathé    |
| PlantDoc    | 17 crops, 27 diseases        | 2,569   | Sing et al.         |
| CDDM        | Multi-crop disease + QA      | 137,000 | CDDM 2024           |

India-specific filter applied: 15 crop classes relevant to AP/Telangana.

---

## Citation

If you use this work, please cite:

```bibtex
@article{kishore2026indagrivlm,
  title   = {IndAgri-VLM: A Telugu-Grounded Multimodal Vision-Language Model
             for South Indian Commercial Crop Disease Diagnosis},
  author  = {Kishore, P.V.V.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
