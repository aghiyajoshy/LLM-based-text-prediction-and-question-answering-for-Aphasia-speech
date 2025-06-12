# LLM-based-text-prediction-and-question-answering-for-Aphasia-speech
This project develops a LLM based system to assist aphasia patients by predicting missing words and answering questions from fragmented or disfluent speech. It combines speech-to-text transcription (AssemblyAI), masked language modeling (RoBERTa), and question answering to reconstruct meaningful sentences and provide relevant responses.
Aphasia Speech Assistant using RoBERTa
This project builds a language model-based system to assist patients with Aphasia — a language disorder that affects speech production and comprehension. The system performs text prediction and question answering on fragmented or disfluent speech using RoBERTa-based models, helping reconstruct meaningful communication.

🚀 Features
✅ Speech-to-Text Transcription using AssemblyAI

✅ Masked Word Prediction using RoBERTa (Masked Language Model)

✅ Question Answering using fine-tuned RoBERTa QA model

✅ Handles disfluencies and speech gaps often seen in aphasia patients

✅ Evaluation Metrics: Accuracy, Precision, F1 Score, BLEU Score

💡 Motivation
Aphasia patients often struggle with missing words, fragmented speech, or incomplete sentences. Traditional speech analysis tools are not designed for such irregular input. This project leverages modern Large Language Models (LLMs) to:

Reconstruct missing words

Understand patient intent

Answer questions from partial speech

Assist therapy and communication

🔧 Technologies Used
RoBERTa-base (Masked Language Model)

RoBERTa-base-SQuAD2 (Question Answering Model)

Hugging Face Transformers

PyTorch

AssemblyAI (Speech-to-Text API)

Python 3.8+

📊 Evaluation Metrics
Metric	Purpose
Accuracy	Correct full predictions
Precision	Reliability of correct predictions
F1 Score	Balance between precision and recall
BLEU Score	Similarity between predicted and reference text

🗂 Dataset
Custom dataset generated from speech samples with disfluency masking and QA pairs extracted.

The project can easily adapt to real-world speech datasets (e.g. AphasiaBank).

🔨 Installation
1️⃣ Clone this repository:

git clone https://github.com/your-username/aphasia-speech-assistant.git
cd aphasia-speech-assistant

2️⃣ Install required packages:

pip install torch transformers assemblyai datasets

3️⃣ Set up AssemblyAI API key in your code:

aai.settings.api_key = "YOUR_API_KEY"

🧪 Running the Project
Upload speech files (.wav) to transcribe and process.

The system automatically:

Transcribes speech

Extracts patient utterances

Masks disfluencies

Predicts missing words

Answers questions from the transcript

📈 Results
The system achieves good performance in reconstructing sentences from aphasia speech and accurately answering simple context-based questions. Evaluation is done using both strict (accuracy) and flexible (BLEU score) metrics.

📚 References
RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al.)

Hugging Face Transformers

AssemblyAI Speech-to-Text API

Aphasia Research Papers

📌 Future Work
Fine-tune RoBERTa on aphasia-specific speech datasets

Extend model to handle more severe disfluencies

Add real-time speech correction interface

