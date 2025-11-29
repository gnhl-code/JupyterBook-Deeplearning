# Week 1 

Nov 10–16 - Narrative Report 

This week was an important step for our group because we finalized the direction of our research. Our study focuses on classifying lung sounds, specifically identifying healthy breaths and abnormal sounds such as crackles and wheezes. Since lung sounds change over time, we want to build a deep learning system that can handle these time-based patterns. Before choosing our methods, we first checked the quality of our data. We reviewed the ICBHI 2017 Respiratory Sound Database, which contains 6,898 respiratory cycles labeled as normal, crackle, wheeze, or both. We confirmed that the labels were complete, but we also noticed that the dataset is noisy and has fewer abnormal samples than healthy ones. This helped us understand the challenges we need to address in our study.

During our November 15, 2025 meeting, we discussed how to write the background of our study and what our objectives should be. Each member shared ideas, and we also talked about the common problems with traditional auscultation, such as how subtle lung sounds are difficult to detect and how interpretations can differ between practitioners. These issues helped strengthen our rationale for studying automated lung sound classification, especially because unstructured medical data like audio is still not widely explored.

![week1](images/week1.png)

The main task for this phase was to decide what we will test in our experiments. We focused on three parts of the pipeline: the input features, the model architecture, and the hyperparameters. For the input, we agreed to compare Mel-Spectrograms and MFCCs to see which representation works better for lung sound signals. For the model, we will compare a standard TCN with a spiking TCN to investigate whether the spiking mechanism can reduce noise more effectively. For the settings, we will experiment with different hyperparameters, such as kernel size and time steps, because these can affect how the model processes sound. We have not chosen the final values yet, so tuning these settings will be a major part of our work.

By combining these decisions, we finalized our scope: we will compare MFCC vs. Mel-Spectrograms, Standard TCN vs. Spiking TCN, and test different hyperparameter settings at the same time. This plan allows us to understand not only which model performs better, but also why it performs that way under different conditions. Overall, this phase helped us move from a broad idea to a clear and organized plan. 


