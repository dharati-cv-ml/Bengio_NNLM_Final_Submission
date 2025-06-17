# Bengio_NNLM_Final_Submission
# Neural Probabilistic Language Model (NNLM) - Bengio et al. (2003)

This repository contains my Python implementation of the Neural Probabilistic Language Model described in:

**Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Janvin. (2003)**  
*A Neural Probabilistic Language Model*. Journal of Machine Learning Research (JMLR).

---

## 🔬 Project Summary

- ✅ Implemented the NNLM architecture fully in PyTorch
- ✅ Preprocessed data using NLTK: tokenization, lemmatization, POS tagging, stopword removal
- ✅ Trained on Brown corpus from NLTK dataset
- ✅ Vocabulary limited to top 10,000 words (`<UNK>` for unknowns)
- ✅ Context size of 4 words (n-gram model)
- ✅ Model trained with cross-entropy loss, Adam optimizer
- ✅ Evaluation performed using perplexity

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- NLTK

Install dependencies:

```bash
pip install torch nltk
