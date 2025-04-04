# MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation

**<div align="center">Preprint</div>**

<div align="center">Khai Le-Duc*, Tuyen Tran*, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh**, Thanh Nguyen-Tang**</div>

<div align="center">*Equal contribution</div>
<div align="center">**Equal supervision</div>

> Please press ⭐ button and/or cite papers if you feel helpful.

* **Abstract:**
Multilingual speech translation (ST) in the medical domain  enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing *MultiMed-ST*, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French,  Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online:  [https://github.com/leduckhai/MultiMed-ST](https://github.com/leduckhai/MultiMed-ST).

* **Citation:**
Please cite this paper: To be on Arxiv

This repository contains scripts for end-to-end automatic speech recognition (ASR), machine translation (MT), and speech translation (ST) using cascaded and end-to-end sequence-to-sequence (seq2seq) models. The provided code covers model preparation, training, inference, and evaluation processes, based on the dataset *MultiMed-ST*.

## Dataset and Pre-trained Models:

Dataset: [HuggingFace dataset](https://huggingface.co/datasets/leduckhai/MultiMed-ST), [Paperswithcodes dataset]()

Fine-tuned models: [HuggingFace models](https://huggingface.co/leduckhai/MultiMed-ST)

## Contact:

Core developers:

**Khai Le-Duc**
```
University of Toronto, Canada
Email: duckhai.le@mail.utoronto.ca
GitHub: https://github.com/leduckhai
```

**Bui Nguyen Kim Hai**
```
Eötvös Loránd University, Hungary
Email: htlulem185@gmail.com
```
