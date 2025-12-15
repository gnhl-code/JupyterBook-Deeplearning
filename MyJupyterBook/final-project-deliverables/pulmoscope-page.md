# **Project PulmoScope**

**A Comparative Study of Hybrid TCN-SNN and Temporal Deep Learning Models for Multi-Class Respiratory Disease Detection from Lung Auscultation Sounds**

_Authors: Genheylou Felisilda, Nicole Menorias, Kobe Marco Olaguir, and Joanna Reyda Santos_

## **Abstract**

Chronic respiratory diseases such as chronic obstructive pulmonary disease (COPD) and pneumonia remain leading causes of morbidity and mortality worldwide, particularly in resource-limited settings where access to advanced diagnostic tools is restricted. Conventional lung auscultation, while widely used, suffers from low sensitivity, high interobserver variability, and difficulty in differentiating diseases with overlapping acoustic signatures. This study presents PulmoScope, a disease-centered deep learning framework for multi-class respiratory disease classification using lung auscultation sounds.

Using the ICBHI 2017 Respiratory Sound Database, lung sound recordings were standardized through rigorous preprocessing, including resampling, bandpass filtering, fixed-duration segmentation, loop-padding, and normalization. A hybrid feature representation combining high-resolution Mel-spectrograms and MFCCs was employed to capture both spectral and cepstral characteristics of respiratory acoustics. To address severe class imbalance, a downsample-then-augment strategy with frequency masking was applied.

The study systematically compares multiple temporal deep learning architectures, including a hybrid Temporal Convolutional Network–Spiking Neural Network (TCN-SNN), a pure TCN, Long Short-Term Memory (LSTM), and a Vanilla RNN, under identical training conditions. Performance was evaluated using accuracy, F1-score, precision, recall, ROC–AUC, and confusion matrix analysis on a held-out test set. Model interpretability was further examined using Grad-CAM to visualize time–frequency regions influencing classification decisions.

Results demonstrate that advanced temporal models, particularly the hybrid TCN-SNN, outperform traditional recurrent baselines in disease-level differentiation, reducing misclassification between COPD and pneumonia. These findings highlight the potential of deep learning–assisted auscultation as a robust clinical decision-support tool for improving early detection and reducing misdiagnosis of respiratory diseases in real-world and resource-constrained healthcare settings.


for more project information heres the github-link: https://github.com/LadyJo02/PulmoScope