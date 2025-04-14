# MusicBERT Genre Classification

## Problem Statement & Overview

This project explores the application of MusicBERT, a transformer-based model for symbolic music understanding, to the task of music genre classification. MusicBERT was initially developed by Microsoft Research for various music understanding tasks, and this implementation adapts it specifically for genre classification.

Music genre classification is a complex task requiring models to identify patterns across rhythm, harmony, instrumentation, and other musical elements. Implementing this with MusicBERT involves navigating various configuration challenges and optimization requirements.

This implementation targets genre classification using MusicBERT with the TOP-MAGD dataset, containing music across 13 different genres. The project demonstrates how to set up and train MusicBERT for this specific classification task.

The implementation includes:
- Configuration of MusicBERT for genre classification tasks
- Implementation of training scripts and environment setup
- Adaptation of the model to handle multi-label classification
- Evaluation using genre classification metrics

This project provides a practical implementation of MusicBERT for music genre classification, focusing on the training pipeline and configuration aspects rather than modifying the core architecture.

## Methodology

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

I trained for **10 epochs** using:
- Batch size: 16
- Max tokens: 2048
- Learning rate: 5e-4
- Warmup: 100 updates
- Optimizer: Adam
- Validation every epoch

The model was saved using **validation accuracy** as the metric, not loss.

