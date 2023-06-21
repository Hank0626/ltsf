# 🚀 LTSF: A Baseline Aggregator for Long-Term Time-Series Forecasting 
[![PyPI version](https://badge.fury.io/py/ltsf.svg)](https://badge.fury.io/py/ltsf)

LTSF is a Python 🐍 package that simplifies the process of implementing and testing various state-of-the-art models 🧠 for long-term time-series forecasting 📈. This package supports the majority of current leading baselines, offering a user-friendly interface to perform LTSF tasks effortlessly.

LTSF enables users to employ their desired model with a single-line configuration and provides the tools 🛠️ to create training, validation, and testing datasets effortlessly. By further utilizing LTSF's training interface, users can train and test their models using only a few lines of code.

## 🔧 Supported Baselines

The table below shows the supported baselines with their corresponding references and links to their original implementation:

| Baseline | Reference | Code |
|---|---|---|
| PatchTST | [ICLR 2023](https://arxiv.org/abs/2211.14730) | [PatchTST](https://github.com/yuqinie98/PatchTST)  |
| MICN     | [ICLR 2023](https://openreview.net/pdf?id=zt53IDUR1U) | [MICN](https://github.com/wanghq21/MICN) |
| FiLM     | [NIPS 2022](https://arxiv.org/abs/2205.08897) | [FiLM](https://github.com/tianzhou2011/FiLM/) |
| TimesNet | [ICLR 2023](https://openreview.net/pdf?id=ju_Uqw384Oq) | [TimesNet](https://github.com/thuml/Time-Series-Library/) |
| Crossformer | [ICLR 2023](https://openreview.net/forum?id=vSVLM2j9eie) | [Crossformer](https://github.com/Thinklab-SJTU/Crossformer) |
| DLinear | [AAAI 2023](https://arxiv.org/pdf/2205.13504.pdf) | [DLinear](https://github.com/cure-lab/LTSF-Linear) |
| LightTS | [arXiv 2022](https://arxiv.org/abs/2207.01186) | [LightTS](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py) |
| ETSformer | [arXiv 2022](https://arxiv.org/abs/2202.01381) | [ETSformer](https://github.com/salesforce/ETSformer) |
| Non-stationary Transformer | [NeurIPS 2022](https://openreview.net/pdf?id=ucNDIDRNjjv) | [Non-stationary Transformer](https://github.com/thuml/Nonstationary_Transformers) |
| FEDformer | [ICML 2022](https://proceedings.mlr.press/v162/zhou22g.html) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| Pyraformer | [ICLR 2022](https://openreview.net/pdf?id=0EXmFzUn5I) | [Pyraformer](https://github.com/ant-research/Pyraformer) |
| Autoformer | [NeurIPS 2021](https://openreview.net/pdf?id=I55UqU-M11y) | [Autoformer](https://github.com/thuml/Autoformer) |
| Informer | [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) | [Informer](https://github.com/zhouhaoyi/Informer2020) |
| Reformer | [ICLR 2020](https://openreview.net/forum?id=rkgNKkHtvB) | [Reformer](https://github.com/lucidrains/reformer-pytorch) |
| Transformer | [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | [Transformer](https://github.com/hyunwoongko/transformer) |


## 📚 Usage

Here is an example of how you can use LTSF:
```
pip install lsf
```

```python
import ltsf

# Create a configuration for your desired model and dataset
config = ltsf.Config("Autoformer", "ETTh1") 

# Set your custom configuration
config.set_config({"use_gpu":False})

# Create training, validation, and testing datasets
train_loader, val_loader, test_loader = ltsf.DatasetFactory.create(config, download=True, data_path=".")  

# Create an LTSFTrainer
trainer = ltsf.LTSFTrainer(config) 

# Start the training process
trainer.train(train_loader, val_loader, test_loader)

# Test the trained model
trainer.test(test_loader, res_dir="./result")
```

## ⭐ Key Features

- Supports 15 leading baselines for long-term time-series forecasting.
- User-friendly, enabling model training and testing with minimal lines of code.

## 🎉 Update Notes
- [2023-06-21] LTSF has now been released! 🚀

## 📈 Future Work
- Planning to support more intelligent parameter settings for improved usability and flexibility.
- Support custom dataset.

## 🙏 Acknowledgments
We are grateful to the authors of all the papers and their original implementations that made this package possible.

## 📜 License
[Apache License](LICENSE).
