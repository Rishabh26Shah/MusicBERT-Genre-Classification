#!/bin/bash

# === PROJECT CONFIGURATION ===
PROJECT_NAME="musicbert_genre_classification"
DATA_DIR="bin_data_genre"
SAVE_DIR="genre_finetune_checkpoints"

# === MODEL CONFIGURATION ===
ARCH="musicbert_small"
NUM_CLASSES=13
CROP_LENGTH=1024

# === TRAINING HYPERPARAMETERS ===
BATCH_SIZE=16
MAX_EPOCH=10
LEARNING_RATE=5e-4
SEED=42
WARMUP_UPDATES=100
WEIGHT_DECAY=0.01
UPDATE_FREQ=2

# === ENVIRONMENT SETUP ===
export disable_cp=True
export mask_strategy=bar
export convert_encoding=OCTMIDI
export crop_length="$CROP_LENGTH"

# === CREATE SAVE DIRECTORY ===
mkdir -p "$SAVE_DIR"

# === TRAINING COMMAND ===
fairseq-train "$DATA_DIR" \
  --user-dir musicbert \
  --task sentence_prediction_multilabel \
  --arch "$ARCH" \
  --num-classes "$NUM_CLASSES" \
  --init-token 0 \
  --separator-token 2 \
  --criterion sentence_prediction_multilabel \
  --max-positions 2048 \
  --max-tokens 2048 \
  --batch-size "$BATCH_SIZE" \
  --update-freq "$UPDATE_FREQ" \
  --optimizer adam \
  --adam-betas '(0.9, 0.999)' \
  --lr "$LEARNING_RATE" \
  --clip-norm 1.0 \
  --save-dir "$SAVE_DIR" \
  --max-epoch "$MAX_EPOCH" \
  --log-interval 10 \
  --train-subset train \
  --valid-subset valid \
  --seed "$SEED" \
  --num-workers 4 \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-updates "$WARMUP_UPDATES" \
  --no-save-optimizer-state \
  --no-epoch-checkpoints \
  --no-last-checkpoints \
  --reset-optimizer \
  --reset-dataloader \
  --reset-meters \
  --best-checkpoint-metric accuracy \
  --maximize-best-checkpoint-metric

