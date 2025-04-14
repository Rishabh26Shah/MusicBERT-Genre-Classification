# MusicBERT Genre Classification

## 1. Problem Statement & Overview

This project explores the application of MusicBERT, a transformer-based model for symbolic music understanding, to the task of music genre classification. MusicBERT was initially developed by Microsoft Research for various music understanding tasks, and this implementation adapts it specifically for genre classification.

Music genre classification is a complex task requiring models to identify patterns across rhythm, harmony, instrumentation, and other musical elements. Implementing this with MusicBERT involves navigating various configuration challenges and optimization requirements.

This implementation targets genre classification using MusicBERT with the TOP-MAGD dataset, containing music across 13 different genres. The project demonstrates how to set up and train MusicBERT for this specific classification task.

The implementation includes:
- Configuration of MusicBERT for genre classification tasks
- Implementation of training scripts and environment setup
- Adaptation of the model to handle multi-label classification
- Evaluation using genre classification metrics

This project provides a practical implementation of MusicBERT for music genre classification, focusing on the training pipeline and configuration aspects rather than modifying the core architecture.

## 2. Methodology

The model I used is `MusicBERT-small`, which is a transformer encoder pretrained on symbolic music using a masked token prediction task. It was originally designed for multiple music understanding tasks. I fine-tuned it for **multi-label genre classification**.

### Input Encoding

Instead of processing raw MIDI data directly, I used MusicBERT’s **OctupleMIDI** encoding. This format turns each musical note into 8 elements, including:

- Time signature
- Tempo
- Bar number
- Position in bar
- Instrument
- Pitch
- Duration
- Velocity

This makes sequences shorter and keeps all the important musical structure. It also helps the model learn patterns across notes, instruments, and timing.

### Masking Strategy

I used **bar-level masking**, where groups of tokens from the same bar are masked together. This was proposed in the MusicBERT paper to avoid leaking information across nearby tokens (since musical patterns often repeat within a bar).

### Model

I used the `musicbert_small` version:
- 4 transformer layers
- 512 hidden size
- 8 attention heads

For the classification task, a simple dense layer was added on top of the model to predict genres. Since this is a **multi-label problem** (songs can belong to multiple genres), I used a multilabel classification head.

### Training

Training configuration:
- Batch size: 16
- Learning rate: 5e-4 (constant)
- Optimizer: Adam
- Max tokens per batch: 2048
- Warmup steps: 100


The model was saved using **validation accuracy** as the metric, not loss.

## 3. Implementation & Demo

This project builds upon the official [MusicBERT](https://github.com/microsoft/muzic/tree/main/MusicBERT) codebase, adapting it for symbolic music genre classification using the TOPMAGD dataset within the Fairseq framework.

### Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/microsoft/muzic.git
cd muzic/MusicBERT

python -m venv musicbert38
source musicbert38/bin/activate

pip install torch fairseq
```

### Configuration Setup

Set environment variables used for preprocessing and training:

```bash
export disable_cp=True
export mask_strategy=bar
export convert_encoding=OCTMIDI
export crop_length=1024
```

### Dataset Preparation

Download and process the TOPMAGD genre classification dataset:

```bash
# Download LMD dataset if not available
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzvf lmd_full.tar.gz
zip -r lmd_full.zip lmd_full

# Download genre mapping file
wget https://raw.githubusercontent.com/andrebola/patterns-genres/master/data/midi_genre_map.json

# Generate dataset in OctupleMIDI format
python -u gen_genre.py
# Follow prompts: 
# subset: topmagd
# LMD dataset zip path: lmd_full.zip
# sequence length: 1000

# Binarize for training
bash binarize_genre.sh topmagd
```

### Training

Run the fine-tuning script to train MusicBERT on genre classification:

```bash
bash train_genre.sh
```

Or use the direct Fairseq command:

```bash
fairseq-train bin_data_genre \
  --user-dir musicbert \
  --task sentence_prediction_multilabel \
  --arch musicbert_small \
  --max-positions 2048 \
  --max-tokens 2048 \
  --batch-size 16 \
  --update-freq 2 \
  --optimizer adam \
  --lr 5e-4 \
  --warmup-updates 100 \
  --criterion sentence_prediction_multilabel \
  --num-classes 13 \
  --init-token 0 \
  --separator-token 2 \
  --save-dir genre_finetune_checkpoints \
  --max-epoch 10 \
  --log-interval 10
```
## 4. Assessment & Evaluation

The model was evaluated on a validation set using metrics tailored for multi-label classification. These include both threshold-based and ranking-based scores to capture the nuances of genre prediction from symbolic music data.

**Key Evaluation Metrics:**
- **Accuracy**: Measures the proportion of correctly predicted genre labels.
- **F1 Score (Micro, Macro, Weighted)**: Evaluates the balance between precision and recall across genres, which is especially important given the class imbalance.
- **ROC AUC (Micro)**: Assesses the model’s ability to rank correct genre labels higher than incorrect ones across all samples.

**Best Performance:**
- **Accuracy**: 61.4%
- **ROC AUC (Micro)**: 0.8756
- **F1 Score (Micro)**: 0.6132

  <img src="https://raw.githubusercontent.com/Rishabh26Shah/MusicBERT-Genre-Classification/main/Score.png" width="800"/>


**What this means**:  
The model was able to correctly identify the relevant genres for around 61% of the samples. The high ROC AUC (micro) indicates that even when predictions weren’t perfectly aligned, the model was still consistently ranking correct genres above incorrect ones. The F1 scores further show balanced performance across precision and recall, despite genre imbalance. The saved checkpoint corresponds to the highest accuracy during validation.

### Note on Results

While MusicBERT has shown strong performance on various symbolic music understanding tasks, in this implementation the fine-tuned genre classification model achieved performance **slightly below** the reported baseline.

Specifically, the validation F1 score was **~0.01 lower** than the baseline in the original MusicBERT paper. This may be due to several factors, including differences in preprocessing, smaller training budget, lack of full hyperparameter tuning, or resource constraints during fine-tuning.

Despite this, the model still demonstrated consistent learning behavior and was able to capture genre-relevant information from symbolic sequences. Further tuning or longer training could likely help close this small performance gap.


## 5. Model & Data Cards

### Model Information

This project uses the `musicbert_small` model, a 4-layer transformer encoder pretrained on symbolic music using masked modeling. Key configuration details:

- **Architecture**: Transformer Encoder (MusicBERT-small)
- **Layers**: 4
- **Hidden Size**: 512
- **Attention Heads**: 8
- **Tokenization**: OctupleMIDI
- **Masking Strategy**: Bar-level masking

This version is suited for fine-tuning on genre classification and other symbolic music understanding tasks.

---

### Intended Uses & Licenses

- **Intended Use**: Symbolic music genre classification
- **Source Model License**: [MIT License](https://opensource.org/licenses/MIT)  
  This includes both the MusicBERT codebase and this adaptation for genre classification.
- **Training & Evaluation Dataset**: [TOPMAGD](https://github.com/andrebola/patterns-genres)  
  This dataset is used to train and evaluate multi-label genre predictions from symbolic MIDI sequences.

---

### Ethical Considerations & Bias

- **Genre Imbalance**: The dataset used in this project contains a skewed distribution across genres. As a result, models might favor the most represented genres unless specifically countered with weighted loss functions or resampling.
- **Music Source Bias**: The underlying MIDI files come from the Lakh MIDI Dataset, which is heavily weighted towards Western, English-language music. As such, predictions may generalize poorly to underrepresented musical traditions or genres.
- **Transparency**: The model does not make decisions based on audio signals or lyrics; it only uses symbolic music structure like pitch, duration, instrumentation, etc.

The implementation is research-oriented and not intended for commercial deployment without additional fairness or robustness evaluation.

## 6. Critical Analysis

This project investigates the use of MusicBERT for symbolic music genre classification. MusicBERT is a transformer-based model pretrained using masked modeling on a large symbolic music dataset with structural encoding and bar-level masking. The main goal was to fine-tune it for a downstream task and evaluate its ability to transfer knowledge.

### What is the impact of this project?  
It shows that pretrained symbolic music models can be adapted to classification tasks with minimal changes. This lowers the barrier to applying deep learning in symbolic music analysis and makes it easier to use models like MusicBERT for tasks like genre tagging or music recommendation.

### What does it reveal or suggest?  
The results suggest that token-based representations and bar-level structure help the model capture genre-relevant features. The model performed consistently and showed that pretrained representations already contain useful information for tasks beyond masked prediction.

### What is the next step?  
Possible future directions include:
- Using larger MusicBERT variants or longer input sequences
- Addressing class imbalance through sampling or loss weighting
- Applying this setup to other tasks such as mood classification or composer identification
- Combining symbolic input with audio features in a hybrid model


## 7. Documentation & Resource Links

### Repo & ReadMe

This repository includes:
- A complete README with setup instructions, usage guide, and training pipeline.
- Training and preprocessing scripts (`train_genre.sh`, `binarize_genre.sh`) for genre classification.
- Code for adapting MusicBERT to the multi-label classification task.
- Environment configuration and reproducibility instructions.

### Resource Links

- [MusicBERT Paper (ACL 2021)](https://arxiv.org/pdf/2106.05630.pdf): *Symbolic Music Understanding with Large-Scale Pre-Training*
- [Official MusicBERT GitHub Repository](https://github.com/microsoft/muzic/tree/main/musicbert)
- [TOPMAGD Dataset & Genre Mapping](https://github.com/andrebola/patterns-genres)
- [Fairseq Framework](https://github.com/facebookresearch/fairseq)

All relevant references and licenses (MIT License) from the original MusicBERT repo are maintained in this implementation.



