# Phase 4 - Narrative Report

Dec 1-7 - Narrative Report 

Phase 4 marked the point where our project moved from experimentation into full, structured implementation. After completing the earlier stages—data preparation, feature engineering, and initial model testing—we focused on building a diagnostic pipeline that was not only functional, but also practical for real-world use, particularly within the Filipino healthcare setting. This phase centered on refining our models, validating their performance, and integrating explainability tools to ensure that the final system was both accurate and trustworthy.

## **Feature Integration and Data Partitioning**
To strengthen the system’s diagnostic capability, we adopted a hybrid feature extraction strategy using both MFCCs and Mel Spectrograms. This gave our models access to a more complete picture of lung sound characteristics, from tonal qualities to temporal shifts. We paired this with a 60/20/20 Train–Validation–Test split, ensuring proper separation of training signals and preventing the models from overfitting to noise or recurring sound patterns in the dataset.

## **Shift From Sound Events to Actual Disease Labels**
Rather than classifying generic respiratory events like wheezes or crackles, we shifted the framework to predict disease-specific labels: COPD, Pneumonia, Healthy, and Other. This adjustment aligned the system with clinical needs. In many Filipino settings, simply hearing “wheezes” or “crackles” is not enough to confidently diagnose a condition. Disease-level classification made the output more actionable and reduced the risk of misinterpreting ambiguous respiratory sounds.

## **Refined Preprocessing: Enhancement Over Augmentation**
While augmentation is commonly used to expand and diversify datasets, our early analysis showed that the lung sound recordings were inherently noisy. Adding further manipulated noise risked weakening the model’s ability to distinguish real pathologies. Instead, we focused on signal enhancement, improving clarity without distorting the natural characteristics of respiratory sounds.

## **Model Implementation and Architecture Evaluation**
Phase 4 also involved the full implementation of the four candidate architectures:
- Vanilla RNN
- LSTM
- Pure TCN
- TCN-SNN

To ensure fairness, each model used the same Shared Deep Classifier Head, preventing performance discrepancies caused by unequal classifier complexity. This allowed us to judge the models purely by their ability to extract meaningful temporal features.
Early Stopping with a patience of 10 was integrated to avoid unnecessary training and detect when the models began to overfit.
Through this process, two models emerged as **top performers: Pure TCN and TCN-SNN**, both demonstrating strong diagnostic behavior and stable convergence during training.

## **Explainability Integration (Grad-CAM)**
To make the system transparent and clinically reliable, we implemented Grad-CAM heatmaps on top of the spectrograms. This allowed us to visualize what parts of the signal the model focused on:
- For Pneumonia, the model consistently highlighted sharp, high-frequency crackle spikes.
- For COPD, it focused on long, continuous harmonic bands associated with wheezes.
- Background noise, silence, and non-diagnostic intervals were assigned minimal attention, confirming that the model learned to ignore irrelevant artifacts.

This explainability layer helped verify that our models were “listening” to medically meaningful cues rather than memorizing noise patterns.

Phase 4 established the backbone of our diagnostic system. We moved from conceptual design into a fully implemented pipeline capable of processing real respiratory signals, classifying diseases accurately, and justifying its decisions through visual explanations. With refined preprocessing, standardized model evaluation, and robust explainability methods, this phase laid the groundwork for reliable deployment and future model improvements.
