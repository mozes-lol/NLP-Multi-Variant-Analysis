# NLP Multi-Variant Text Analysis (BERT, GPT, and Text-GAN)

## Project Overview

This repository contains the implementation of three distinct Natural Language Processing (NLP) deep learning architectures to process, classify, and generate text based on real-world customer feedback.

The objective of this project is to evaluate and compare the operational mechanics, performance limits, and behavioral differences of:

1. **Language Representation (BERT)** - Sequence Classification (Star Rating Prediction)
2. **Language Generation (GPT)** - Causal Language Modeling (Review Generation)
3. **Adversarial Text Generation (GAN)** - Synthetic Text Generation

## Dataset Repository

**McDonald's Store Reviews Dataset** Source: [Kaggle - nelgiriyewithana/mcdonalds-store-reviews](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)

The corpus consists of over 33,000 anonymized customer reviews scraped from Google Reviews covering various McDonald's locations. It features a rich mix of formal feedback, colloquial phrasing, grammatical errors, and emojis, making it an excellent testing ground for NLP architectures.

## Performance Evaluation Matrix

| Model Variant | Primary Metric (Precision / Recall / F1) | Generative Quality Metric (BLEU / ROUGE / Perplexity) | Training Time per Epoch | Key Observations / Constraints | 
| :--- | :--- | :--- | :--- | :--- | 
| **BERT Variant** (bert-base-uncased) | P: 0.6275<br>R: 0.6084<br>F1: 0.6125 | N/A (Classification Task) | ~254s | Highly effective at sentiment classification. Bidirectional context helps identify nuances in star ratings. |
| **GPT Variant** (DistilGPT2) | N/A (Generative Task) | BLEU: 0.0279<br>ROUGE-L: 0.2154<br>PPL: 16.66 | ~150s | Excellent fluency and coherence. Captures the "voice" of McDonald's customers effectively. Low BLEU/ROUGE is expected due to the creative nature of autoregressive generation. |
| **Text-GAN Variant** (Custom 1D-CNN) | D-Acc: 0.3934 | N/A (Poor Quality) | ~5s | Struggled with discrete sequence generation. Fast to train per batch but requires complex Reinforcement Learning (RL) logic to achieve true semantic coherence. |

## Analytical Discussion

### 1. Tokenization Differences
The choice of tokenization fundamentally altered how each model processed the dataset:
* **BERT:** Utilizes **WordPiece tokenization**, which is optimized for sub-word understanding. It effectively handled typos and root words in the reviews, mapping them to dense embeddings.
* **GPT:** Utilizes **Byte-Pair Encoding (BPE)**. This allowed the autoregressive model to efficiently predict the next token by breaking down colloquial fast-food phrasing into frequent character pairs, maintaining the informal "voice" of the dataset.
* **Text-GAN:** Utilized a simpler **word-level Tokenizer** (Keras Tokenizer). This significantly increased the output layer dimensionality (5000+ nodes representing the vocabulary) compared to the dense embeddings of the Transformers, making the generator's prediction task exponentially harder.

### 2. Metric Tradeoffs & Architectural Limits (Why GANs Struggle)
While Generative Adversarial Networks (GANs) are the gold standard for continuous data (like image synthesis), they inherently struggle with NLP. 
* **The Discrete Text Problem:** Text generation is a discrete process (you output a specific word index, not a continuous pixel value). Because the output is non-differentiable, the discriminator cannot easily pass gradients back to the generator to update its weights during standard backpropagation.
* **Lack of Pre-training:** Transformers (BERT/GPT) benefit from pre-training on massive corpora, allowing them to excel with relatively little fine-tuning. The GAN, built from scratch, lacks this "world knowledge". 
* **Conclusion:** Autoregressive (GPT) and Encoder (BERT) architectures remain vastly superior for NLP tasks due to their ability to model long-range dependencies via the Attention mechanism, whereas the Text-GAN requires highly complex Reinforcement Learning frameworks (like SeqGAN policy gradients) to generate anything beyond repetitive, disjointed words.

## Repository Contents
* `notebook.ipynb` - The complete code pipeline containing data preprocessing, model initializations, training loops, and test inferences.
* `notebook.pdf` - PDF export of the completed notebook.
