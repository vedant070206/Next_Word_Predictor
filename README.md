# Shakespearean Next-Word Prediction with LSTM

This project presents an end-to-end LSTM-based natural language processing system for next-word prediction. The model is trained on the English of William Shakespeare, especially the language used in Hamlet, and is designed to generate or suggest words and phrases that reflect old English and Shakespearean style.

## Overview

This model learns patterns from Shakespearean text and predicts the most likely next word for a given input phrase. Because it was trained on archaic English, it performs best when the input text resembles Shakespearean or early modern English. The project includes a Streamlit web app for interactive prediction.

## Dataset

The training data is sourced from Shakespeare's Hamlet text stored in the repository as hamlet.txt.

### Data Description
- Source: Shakespeare's play Hamlet
- Language Style: Early Modern English / Shakespearean English
- Content Type: Dialogue and poetic monologues
- Purpose: Teach the model word-by-word context and stylistic patterns of old English
- Preprocessing: Text was cleaned, tokenized, converted into word sequences, and prepared for LSTM training

## Training Metrics

The best-performing checkpoint was observed during training with the following metrics:

| Metric | Value |
|--------|-------:|
| Training Loss | 6.5242 |
| Training Accuracy | 0.0393 |
| Training Top-5 Accuracy | 0.1286 |
| Validation Loss | 6.6437 |
| Validation Accuracy | 0.0466 |
| Validation Top-5 Accuracy | 0.1372 |

> These results reflect the model’s early-stage learning behavior on Shakespearean text and highlight the challenge of predicting the next word in a highly stylistic and historical language.

## Live Demo

Try the interactive web app here:

- Streamlit App: https://nextwordpredictor-35xdkvvtzpttn9bgljzccm.streamlit.app/

## Model Details

- Model Type: Long Short-Term Memory (LSTM) neural network
- Task: Next-word prediction
- Output: Top predicted next words with confidence scores
- Deployment: Interactive Streamlit application in app.py

## Author

- Name: Vedant Thaker
- LinkedIn: https://www.linkedin.com/in/vedant-thakar-4ba561292/
- GitHub: https://github.com/vedant070206