# Zemberek_NaiveBayes_NLP
Turkish Triage Classification System with Zemberek 
Naive BayesThis project was developed as a final assignment for the Introduction to Natural Language Processing (NLP) course. The project has been successfully completed and presented as part of the academic requirements.
Project Overview
The "Turkish Triage Classification System" is an NLP-based decision support tool designed for emergency rooms. It automatically analyzes patient complaints written in Turkish and classifies them into two categories: Emergency or Non-Emergency. The primary goal is to minimize human-induced delays in medical prioritization and assist healthcare professionals in high-pressure environments.
Key Features
1) Morphological Analysis: Integrated Zemberek library to handle the complex agglutinative structure of the Turkish language.

2) Advanced Vectorization: Utilizes TF-IDF with N-Gram (1, 3) logic to capture contextual meaning (e.g., distinguishing between "minor pain" and "heart pain").

3) High Performance: Achieved a 92% overall accuracy using the Multinomial Naive Bayes algorithm.

4) Real-time Inference: A command-line interface for instant triage prediction based on user input.

Methodology & Pipeline
Preprocessing: Raw text is tokenized and lemmatized using Zemberek to reduce sparsity (e.g., converting "kanaması", "kanıyor" to the root "kana-").

Feature Extraction: Words are converted into numerical vectors using TF-IDF, emphasizing medical keywords like "bilinç" (consciousness) or "kalp" (heart).

Modeling: The Multinomial Naive Bayes model was trained on a labeled dataset of medical complaints.


Results
The model demonstrates robust performance in distinguishing critical cases:

Accuracy: 92%

Emergency Recall: 89%

Non-Emergency Recall: 94%

Example Case:

Input: "Kalbim sıkışacak ölecek gibi hissediyorum" (I feel like my heart is squeezing and I'm going to die) Output: Emergency (Confidence: 96.35%)
