# 🩺 **MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation**

<p align="center">
  <a href="https://arxiv.org/abs/2504.03546">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2504.03546-b31b1b?logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/leduckhai/MultiMed-ST">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-blue?logo=huggingface&logoColor=white" alt="Dataset">
  </a>
  <a href="https://huggingface.co/leduckhai/MultiMed-ST">
    <img src="https://img.shields.io/badge/Models-HuggingFace-green?logo=huggingface&logoColor=white" alt="Models">
  </a>
  <a href="https://github.com/leduckhai/MultiMed-ST/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </a>
  <a href="https://github.com/leduckhai/MultiMed-ST/stargazers">
    <img src="https://img.shields.io/github/stars/leduckhai/MultiMed-ST?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <strong>📘 EMNLP 2025</strong>
</p>

<p align="center">
  <b>Khai Le-Duc*</b>, <b>Tuyen Tran*</b>, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh**, Thanh Nguyen-Tang**
</p>

<p align="center">
  <sub>*Equal contribution &nbsp;&nbsp;|&nbsp;&nbsp; **Equal supervision</sub>
</p>

---

> ⭐ **If you find this work useful, please consider starring the repo and citing our paper!**

---

## 🧠 Abstract

Multilingual speech translation (ST) in the **medical domain** enhances patient care by enabling effective communication across language barriers, alleviating workforce shortages, and improving diagnosis and treatment — especially in global health emergencies.

In this work, we introduce **MultiMed-ST**, the *first large-scale multilingual medical speech translation dataset*, spanning **all translation directions** across **five languages**:  
🇻🇳 Vietnamese, 🇬🇧 English, 🇩🇪 German, 🇫🇷 French, 🇨🇳 Traditional & Simplified Chinese.

With **290,000 samples**, *MultiMed-ST* represents:
- 🧩 the **largest medical MT dataset** to date  
- 🌐 the **largest many-to-many multilingual ST dataset** across all domains  

We also conduct the **most comprehensive empirical analysis** in ST research so far, covering:
- ✅ Empirical baselines  
- 🔄 Bilingual vs. multilingual study  
- 🧩 End-to-end vs. cascaded models  
- 🎯 Task-specific vs. multi-task seq2seq approaches  
- 🗣️ Code-switching analysis  
- 📊 Quantitative & qualitative error analysis  

All **code, data, and models** are publicly available:  
👉 [**GitHub Repository**](https://github.com/leduckhai/MultiMed-ST)

---

## 🧰 Repository Overview

This repository provides scripts for:

- 🎙️ **Automatic Speech Recognition (ASR)**
- 🌍 **Machine Translation (MT)**
- 🔄 **Speech Translation (ST)** — both **cascaded** and **end-to-end** seq2seq models  

It includes:

- ⚙️ Model preparation & fine-tuning  
- 🚀 Training & inference scripts  
- 📊 Evaluation & benchmarking utilities  

---

## 📦 Dataset & Models

- **Dataset:** [🤗 Hugging Face Dataset](https://huggingface.co/datasets/leduckhai/MultiMed-ST)  
- **Fine-tuned Models:** [🤗 Hugging Face Models](https://huggingface.co/leduckhai/MultiMed-ST)

---

## 👨‍💻 Core Developers

1. **Khai Le-Duc**  

University of Toronto, Canada 

📧 [duckhai.le@mail.utoronto.ca](mailto:duckhai.le@mail.utoronto.ca)  
🔗 [https://github.com/leduckhai](https://github.com/leduckhai)

2. **Tuyen Tran**: 📧 [tuyencbt@gmail.com](mailto:tuyencbt@gmail.com) 

Hanoi University of Science and Technology, Vietnam

3. **Nguyen Kim Hai Bui**: 📧 [htlulem185@gmail.com](mailto:htlulem185@gmail.com)  

Eötvös Loránd University, Hungary 

## 🧾 Citation

If you use our dataset or models, please cite:

📄 [arXiv:2504.03546](https://arxiv.org/abs/2504.03546)

```bibtex
@inproceedings{le2025multimedst,
  title={MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation},
  author={Le-Duc, Khai and Tran, Tuyen and Tat, Bach Phan and Bui, Nguyen Kim Hai and Anh, Quan Dang and Tran, Hung-Phong and Nguyen, Thanh Thuy and Nguyen, Ly and Phan, Tuan Minh and Tran, Thi Thu Phuong and others},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={11838--11963},
  year={2025}
}
