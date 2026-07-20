# AAAI2026-AEMM
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/pdf?id=uXKgVqYTJ2) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![AAAI](https://img.shields.io/badge/AAAI-2026-purple.svg) 

The official implementation for **"[Learning Adaptive and Expandable Mixture Model for Continual Learning](##)"** (AAAI2026) 

------

## ▶️ Usage

### **1. Create env and install requirements**

```bash
conda create -n AEMM python=3.10
conda activate AEMM
pip install -r requirements.txt
```

### **2. Run the example training script**

```bash
bash AEMM.sh
```

### Project structure overview

```bash
LEAR/
├── backbone/                 # Pre-trained backbone models
│   ├── AEMM.py               # LEAR backbone implementation
│   └── ...
├── datasets/                 # Dataset loaders
|   ├── init.py               # Modify domain sequence                
│   └── ...
├── models/                   # CL Method implementations
│   └── AEMM.py               # LEAR method implementation
├── utils/                    # Helper tools
|   ├── train_domain.py       # Training scripts                
│   └── ...
├── main_domain.py            # Main entry
├── AEMM.sh
└── README.md
```

------

## 📝 Citation

If you find this repository helpful, please click the ⭐Star and cite our paper:

```
@inproceedings{ye2026learning,
  title={Learning adaptive and expandable mixture model for continual learning},
  author={Ye, Fei and Zhong, YongCheng and Liu, Qihe and Bors, Adrian G and Sun, JingLing and Guo, JinYu and Zhou, ShiJie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={33},
  pages={27773--27781},
  year={2026}
}
```

------

## 🙏 Acknowledgement

Thanks for the awesome continual learning framework **[Mammoth](https://github.com/aimagelab/mammoth)**.
