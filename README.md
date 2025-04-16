# **Scaling Down Without Losing Accuracy: Sentiment Analysis with Knowledge Distillation and LoRA**

## **Getting Started**
This project explores efficient sentiment analysis by reducing the size of large transformer models while maintaining high accuracy. Using Knowledge Distillation (KD) and Low-Rank Adaptation (LoRA), we optimize transformer-based models for faster inference and lower computational cost while preserving performance.

---

## **Contributions**
- Yashwanthsai Pathipati (U58581488) - performing knowledge distillation and benchmarking the performance and **Robustness** of the distilled model on an unseen dataset
- Himasree Pathipati (U56299154) - Fine-tuning teacher model and benchmarking the performance and **Robustness** of the model on an unseen dataset
- Srichandana Rangula (U54607515) - Fine-tuning student model and benchmarking the performance and **Robustness** of the model on an unseen dataset

---

## **Motivation**
Transformer-based models like BERT achieve state-of-the-art accuracy in NLP tasks but suffer from:

- **High computational cost** – Requires powerful GPUs for training and inference.
- **Slow inference time** – Not ideal for real-time applications.
- **Limited deployment feasibility** – Difficult to run on edge devices.

To address these challenges, our project focuses on:
- **Knowledge Distillation**: Transferring knowledge from a larger model (BERT) to a smaller one (DistilBERT).
- **LoRA Optimization**: Efficient fine-tuning technique to reduce memory usage.
- **Maintaining Accuracy**: Ensuring minimal performance loss despite compression.

---

## **Key Features**
- Fine-tuned sentiment analysis model for efficient text classification. 
- Uses Knowledge Distillation (KD) to compress model size while retaining performance. 
- LoRA (Low-Rank Adaptation) optimization for memory-efficient fine-tuning. 
- Benchmarking and comparative analysis with performance evaluation metrics.

---

## **Technologies Used**
- Deep Learning Frameworks: PyTorch, Hugging Face Transformers 
- Trustworthy Principle: Robustness and Reliability
- Dataset: Stanford Sentiment Treebank (SST-2)
- Benchmarking dataset: stanfordnlp/imdb
- Optimization Techniques: Knowledge Distillation (KD), LoRA 
- Performance Evaluation: Accuracy, F1-score, Inference Time, Memory Usage
- Training Environment: Google Colab Pro
- ChatGPT: For guidance and support

---

## **Trustworthiness Evaluation Metric**
We chose robustness and reliability as the core trustworthy AI principles to evaluate my project because they directly reflect a model's stability and consistent performance under unseen or challenging data conditions. Ensuring that the sentiment analysis model remains accurate and dependable, even after compression through knowledge distillation, is crucial for real-world applicability.

---

## **Project Structure**
```
/sentiment-analysis
├── Sentiment_Analysis.ipynb   # Jupyter Notebook with code implementation
├── requirements.txt           # Dependencies for running the project
├── README.md                  # Project documentation
└── results                    # Evaluation metrics and visualizations

```

## **Installation & Setup**
1. Clone the repository:

```
git clone https://github.com/PathipatiYashwanthSai/Sentiment-Analysis-Using-Knowledge-Distillation.git
cd Sentiment-Analysis-Using-Knowledge-Distillation
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```
jupyter notebook Sentiment_Analysis.ipynb
```

- Note: Make sure you have a wandb API key for training the models
---

## **Evaluation Metrics**
- **Accuracy**: Measures correct sentiment classification.
- **F1-Score**: Balances precision and recall for performance evaluation.
- **Inference Time**: Assesses model speed in real-time applications.
- **Memory Usage**: Evaluates efficiency in resource-limited environments.

---

## **Results**

|         Model         | Accuracy | F1-Score | Precision | Recall |
|:---------------------:|:--------:|:--------:|:---------:|:------:|
|    BERT Fine-Tuned    |   91%    |   91%    |    90%    |  92%   |
| DistilBERT Fine-Tuned |   83%    |   83%	   |    84%    |  82%   |
|     DistilBERT KD     |   90%    |   91%    |    89%    |  92%   |

---

## **Benchmarking Results**

|         Model         | Accuracy | F1-Score | Precision | Recall |
|:---------------------:|:--------:|:--------:|:---------:|:------:|
|    BERT Fine-Tuned    |   86%    |   86%    |    86.9%    |  86%   |
| DistilBERT Fine-Tuned |   83.1%    |   82.9%	   |    84.7%    |  83.1%   |
|     DistilBERT KD     |   88.7%    |   88.7%    |    88.9%    |  88.7%   |

---

## **Key Findings**

- LoRA optimization reduces memory usage while maintaining high accuracy.
- Knowledge Distillation significantly improves **DistilBERT's** performance.
- The final **distilled model is 52% faster and half the memory size** than BERT with only a slight accuracy drop.

---

## **References**
- Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS. 
- Hu, E., Wang, Y., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv.
- Devlin, J., Chang, M.-W., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT.
- Sanh, V., Debut, L., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv.
- ⁠Zhao, X., Huang, W., Banks, A., Cox, V., Flynn, D., Schewe, S., & Huang, X. (2021). "Assessing the Reliability of Deep Learning Classifiers Through Robustness Evaluation and Operational Profiles"
- Nalbandyan, G., Shahbazyan, R., & Bakhturina, E. (2025). "SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models"

---
