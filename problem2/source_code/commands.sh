#!/usr/bin/env bash
# =============================================================================
# commands.sh — Problem 2: Character-Level Name Generation (RNN Variants)
# CSL 7640 - Natural Language Understanding, Assignment 2
# Author: B23CS1061
#
# Run this file or copy-paste individual sections to re-run any step.
# All commands assume you are inside problem2_rnn/ as the working dir.
# Activate the venv first:  source ../venv311/bin/activate
# =============================================================================

cd "$(dirname "$0")"    # make sure we're in problem2_rnn/
source ../venv311/bin/activate

# =============================================================================
# STEP 1 — Train all three models
#
# Models:
#   VanillaRNN   : 2-layer Elman RNN, hidden=256  (223K params)
#   BLSTM        : ELMo-style BiLM, 2-layer LSTM×2, hidden=128 (472K params)
#   AttentionRNN : 2-layer Elman RNN + Bahdanau attention, hidden=256 (362K params)
#
# Key design decisions:
#   - BLSTM training uses joint (fwd+bwd) CE loss; validation/checkpoint uses
#     forward CE only, making val_loss directly comparable across all models
#   - BLSTM hidden_size=128 so two LSTMs stay param-comparable to single-RNN models
#   - AttentionRNN uses nn.RNN (plain Elman, tanh) — NOT GRU/LSTM
#
# Reads:   data/TrainingNames.txt
# Writes:  models/VanillaRNN.pt   models/BLSTM.pt   models/AttentionRNN.pt
#          outputs/all_loss_curves.png
#          outputs/VanillaRNN_loss_curve.png
#          outputs/BLSTM_loss_curve.png
#          outputs/AttentionRNN_loss_curve.png
#          outputs/training_summary.json
#          outputs/vocab.json
#
# Results (2026-03-25, Apple MPS device):
#   VanillaRNN   — 223,325 params | best_val=1.9803 | time=12.2s
#   BLSTM        — 472,186 params | best_val=1.9919 | time= 9.1s
#   AttentionRNN — 362,077 params | best_val=1.9869 | time=30.1s
# =============================================================================
python train.py

# =============================================================================
# STEP 2 — Evaluate all three models (Task-2 & Task-3)
#
# Generates 200 names per model, computes:
#   Novelty Rate : % of generated names NOT in the training set
#   Diversity    : unique generated names / total × 100
#   Avg Length   : mean character length
#
# Also writes a qualitative analysis report (Task-3):
#   - Architecture descriptions and realism assessment
#   - Common failure modes per model
#   - Implementation correction notes
#
# Reads:   models/*.pt  +  outputs/vocab.json  +  data/TrainingNames.txt
# Writes:  outputs/evaluation_results.json
#          outputs/evaluation_report.txt
#          outputs/generated_names_VanillaRNN.txt
#          outputs/generated_names_BLSTM.txt
#          outputs/generated_names_AttentionRNN.txt
#
# Results (2026-03-25):
#   VanillaRNN   — Novelty=93.47%  Diversity=99.5%  AvgLen=6.04
#   BLSTM        — Novelty=94.5%   Diversity=98.5%  AvgLen=5.83
#   AttentionRNN — Novelty=92.0%   Diversity=98.5%  AvgLen=6.09
# =============================================================================
python evaluate.py

# =============================================================================
# FULL PIPELINE — run everything end-to-end
# =============================================================================
# python train.py && python evaluate.py
