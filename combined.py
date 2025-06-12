import os
import pyttsx3
import assemblyai as aai
import torch
import json
import re
import random
from transformers import BertTokenizer, BertForMaskedLM, BertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForQuestionAnswering

# Set AssemblyAI API key
aai.settings.api_key = "56481a160f944ad88be6220e68114dea"

# Initialize transcriber with disfluencies enabled
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(
    disfluencies=True,
    speaker_labels=True
)

# Path to audio file
file_path = r"C:\Users\aghiya\OneDrive\Documents\mod_project\aphasia.wav"

# Transcribe audio
transcript = transcriber.transcribe(file_path, config=config)
transcript_text = transcript.text
print("TRANSCRIPT RECEIVED:")
print(transcript_text)


def process_transcript_for_roberta(transcript, mask_count=5, min_sentence_length=3):
    parts = re.split(r'(?<=[.!?])\s+', transcript)
    patient_sentences = []
    
    for part in parts:
        if not part.strip():
            continue
            
        if '?' in part or any(prompt in part.lower() for prompt in ['tell me', 'could you', 'how about', 'what about']):
            continue
        
        words = part.split()
        if len(words) >= min_sentence_length:
            patient_sentences.append(part)
    
    if hasattr(transcript, 'utterances') and transcript.utterances:
        patient_sentences = []
        for utterance in transcript.utterances:
            if utterance.speaker != "A":
                patient_sentences.append(utterance.text)
    
    masked_sentences = []
    disfluencies = ['uh', 'um', 'em', 'er', 'm', 'hmm', 'ah', 'eh']
    
    for sentence in patient_sentences:
        masked_sentence = sentence
        for disfluency in disfluencies:
            pattern = r'\b' + re.escape(disfluency) + r'\b'
            if re.search(pattern, masked_sentence, re.IGNORECASE):
                masked_sentence = re.sub(pattern, '<mask>', masked_sentence, flags=re.IGNORECASE)
        
        if '<mask>' in masked_sentence and len(masked_sentence.split()) >= min_sentence_length:
            masked_sentences.append(masked_sentence)
    
    if len(masked_sentences) < mask_count:
        for sentence in patient_sentences:
            if len(masked_sentences) >= mask_count:
                break
                
            if any(s for s in masked_sentences if s.startswith(sentence[:20])):
                continue
                
            words = sentence.split()
            if len(words) < min_sentence_length:
                continue
                
            if len(words) > 2:
                idx = random.randint(1, len(words) - 2)
                words[idx] = '<mask>'
                masked_sentences.append(' '.join(words))
    
    if len(masked_sentences) < mask_count:
        print("Not enough sentences with disfluencies. Using original sentences for masking.")
        for sentence in patient_sentences:
            if len(masked_sentences) >= mask_count:
                break
                
            if any(s for s in masked_sentences if s.startswith(sentence[:20])):
                continue
                
            words = sentence.split()
            if len(words) >= min_sentence_length:
                words[len(words) // 2] = '<mask>'
                masked_sentences.append(' '.join(words))
    
    print(f"Found {len(patient_sentences)} patient sentences")
    print(f"Created {len(masked_sentences)} masked sentences for RoBERTa")
    
    return masked_sentences

# Function to process transcript for masked text prediction (BERT version)
def process_transcript_for_bert(transcript, mask_count=5, min_sentence_length=3):
    parts = re.split(r'(?<=[.!?])\s+', transcript)
    patient_sentences = []

    for i, part in enumerate(parts):
        if not part.strip():
            continue
        if '?' in part or any(prompt in part.lower() for prompt in ['tell me', 'could you', 'how about', 'what about']):
            continue
        words = part.split()
        if len(words) >= min_sentence_length:
            patient_sentences.append(part)

    if hasattr(transcript, 'utterances') and transcript.utterances:
        patient_sentences = []
        for utterance in transcript.utterances:
            if utterance.speaker != "A":  
                patient_sentences.append(utterance.text)

    masked_sentences = []
    disfluencies = ['uh', 'um', 'em', 'er', 'm', 'hmm', 'ah', 'eh']

    for sentence in patient_sentences:
        masked_sentence = sentence
        for disfluency in disfluencies:
            pattern = r'\b' + re.escape(disfluency) + r'\b'
            if re.search(pattern, masked_sentence, re.IGNORECASE):
                masked_sentence = re.sub(pattern, '[MASK]', masked_sentence, flags=re.IGNORECASE)

        if '[MASK]' in masked_sentence and len(masked_sentence.split()) >= min_sentence_length:
            masked_sentences.append(masked_sentence)

    if len(masked_sentences) < mask_count:
        for sentence in patient_sentences:
            if len(masked_sentences) >= mask_count:
                break
            if any(s for s in masked_sentences if s.startswith(sentence[:20])):
                continue
            words = sentence.split()
            if len(words) < min_sentence_length:
                continue
            if len(words) > 2:
                idx = random.randint(1, len(words) - 2)
                words[idx] = '[MASK]'
                masked_sentences.append(' '.join(words))

    print(f"Found {len(patient_sentences)} patient sentences")
    print(f"Created {len(masked_sentences)} masked sentences for BERT")
    
    return masked_sentences

# Process transcript and save masked sentences for both models
roberta_masked_sentences = process_transcript_for_roberta(transcript_text)
bert_masked_sentences = process_transcript_for_bert(transcript_text)

roberta_samples_file = r"C:\Users\aghiya\OneDrive\Documents\mod_project\roberta_masked_output.txt"
bert_samples_file = r"C:\Users\aghiya\OneDrive\Documents\mod_project\bert_masked_output.txt"

# Save RoBERTa masked sentences
with open(roberta_samples_file, "w", encoding="utf-8") as file:
    for sentence in roberta_masked_sentences:
        file.write(sentence + "\n")

# Save BERT masked sentences
with open(bert_samples_file, "w", encoding="utf-8") as file:
    for sentence in bert_masked_sentences:
        file.write(sentence + "\n")

# Initialize models for both BERT and RoBERTa
# RoBERTa Models
roberta_mlm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_mlm_model = RobertaForMaskedLM.from_pretrained('roberta-base')
roberta_mlm_model.eval()

roberta_qa_tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
roberta_qa_model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
roberta_qa_model.eval()

# BERT Models
bert_mlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_mlm_model.eval()

bert_qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_qa_model.eval()

# Function for RoBERTa masked word prediction
def roberta_predict_masked_text(text):
    if '<mask>' not in text:
        text = text.replace('[MASK]', '<mask>')
    
    if '<mask>' not in text:
        return "No mask token found in text"
    
    inputs = roberta_mlm_tokenizer(text, return_tensors="pt")
    mask_token_id = roberta_mlm_tokenizer.mask_token_id
    mask_indices = torch.where(inputs['input_ids'][0] == mask_token_id)[0]
    
    if len(mask_indices) == 0:
        return "Mask token not recognized by tokenizer"
    
    with torch.no_grad():
        outputs = roberta_mlm_model(**inputs)
        predictions = outputs.logits
    
    result_text = text
    for mask_idx in mask_indices:
        top_token_ids = torch.topk(predictions[0, mask_idx], 5).indices.tolist()
        top_words = [roberta_mlm_tokenizer.decode([token_id]).strip() for token_id in top_token_ids]
        predicted_word = top_words[0]
        result_text = result_text.replace('<mask>', predicted_word, 1)
    
    return result_text

# Function for BERT masked word prediction with intermediate probability selection
def bert_predict_masked_text(text, index):
    if '[MASK]' not in text:
        return "No mask token found in text"

    inputs = bert_mlm_tokenizer(text, return_tensors="pt")
    mask_token_id = bert_mlm_tokenizer.mask_token_id
    mask_indices = torch.where(inputs['input_ids'][0] == mask_token_id)[0]

    if len(mask_indices) == 0:
        return "Mask token not recognized by tokenizer"

    with torch.no_grad():
        outputs = bert_mlm_model(**inputs)
        predictions = outputs.logits

    result_text = text

    for mask_idx in mask_indices:
        top_values, top_token_ids = torch.topk(predictions[0, mask_idx], 10)
        top_probs = torch.nn.functional.softmax(top_values, dim=-1)
        top_words = [bert_mlm_tokenizer.decode([token_id]).strip() for token_id in top_token_ids.tolist()]

        if index < 5:
            selected_word = top_words[0]  # Most probable word
        elif index < 6:
            selected_word = top_words[1]  # Second most probable word
        else:
            selected_word = top_words[-1]  # Least probable word

        result_text = result_text.replace('[MASK]', selected_word, 1)
    
    return result_text

# Functions for question answering
def roberta_question_answering(question, context):
    inputs = roberta_qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_qa_model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits).item()
    answer_end = torch.argmax(outputs.end_logits).item()
    
    if answer_end < answer_start:
        answer_end = answer_start
    
    answer_tokens = inputs.input_ids[0][answer_start:answer_end + 1]
    answer = roberta_qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer if len(answer.split()) <= 5 else " ".join(answer.split()[:5]) + "..."

def bert_question_answering(question, context):
    inputs = bert_qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_qa_model(**inputs)

    start_indices = torch.argsort(outputs.start_logits, descending=False)
    end_indices = torch.argsort(outputs.end_logits, descending=False)

    middle_index = len(start_indices[0]) // 2
    start_index = start_indices[0].tolist()[middle_index]
    end_index = end_indices[0].tolist()[middle_index]

    if end_index < start_index:
        end_index = start_index

    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    answer = bert_qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer if len(answer.split()) <= 5 else " ".join(answer.split()[:5]) + "..."

# Function for text-to-speech
def text_to_speech(text, output_file):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    engine.save_to_file(text, output_file)
    engine.runAndWait()
    print(f"Speech saved to '{output_file}'")

# Function to play audio
def play_audio(file):
    if os.path.exists(file):
        os.system(f'start {file}')
    else:
        print(f"{file} not found.")

# Function to generate predictions and results for RoBERTa
def generate_roberta_results():
    # Load RoBERTa text samples
    with open(roberta_samples_file, 'r', encoding='utf-8') as file:
        roberta_text_samples = file.read().splitlines()
    
    # Skip the first sample and reorder numbering
    roberta_text_samples = roberta_text_samples[1:] if roberta_text_samples else []
    
    # Output file for RoBERTa text predictions
    roberta_output_file = r"C:\Users\kathr\OneDrive\Documents\mod_project\roberta_text.txt"
    
    # Open file for writing RoBERTa predictions
    with open(roberta_output_file, 'w', encoding='utf-8') as file:
        if not roberta_text_samples:
            file.write("No valid masked sentences found.\n")
        else:
            for i, text in enumerate(roberta_text_samples, start=1):
                if '<mask>' not in text and '[MASK]' not in text:
                    file.write(f"Sample {i}:\nInput: {text}\nNo mask token found, skipping.\n\n")
                    continue
                
                result = roberta_predict_masked_text(text)
                result = result.replace('[MASK]', '').replace('<mask>', '').strip()
                text = text.replace('[MASK]', '').replace('<mask>', '').strip()
                file.write(f"Sample {i}:\nInput: {text}\nPrediction: {result}\n\n")
    
    print(f"RoBERTa predictions written to {roberta_output_file}")
    
    # Load QA pairs and process
    qa_pairs_file = r"C:\Users\kathr\OneDrive\Documents\mod_project\extracted_qa.json"
    try:
        with open(qa_pairs_file, 'r', encoding='utf-8') as file:
            qa_pairs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        qa_pairs = []
    
    # Output file for RoBERTa QA results
    roberta_qa_file = r"C:\Users\kathr\OneDrive\Documents\mod_project\roberta_answers.txt"
    
    print("\nROBERTA QUESTION ANSWERING RESULTS:")
    with open(roberta_qa_file, 'w', encoding='utf-8') as file:
        for i, pair in enumerate(qa_pairs):
             
            result = roberta_question_answering(pair["question"], pair["context"])
    
            file.write(f"Question {i + 1}: {pair['question']}\nPredicted Answer: {result}\n\n")
    
    # Generate audio for RoBERTa text prediction
    with open(roberta_output_file, 'r', encoding='utf-8') as file:
        roberta_text_prediction_results = file.read()
    
    text_to_speech(f"RoBERTa text prediction results are:\n{roberta_text_prediction_results}", "roberta_text.mp3")
    
    # Generate audio for RoBERTa question answering
    with open(roberta_qa_file, 'r', encoding='utf-8') as file:
        roberta_qa_results = file.read()
    
    text_to_speech(f"RoBERTa Question Answering results are:\n{roberta_qa_results}", "roberta_answers.mp3")
    
    print("RoBERTa results generated and saved to files.")

# Function to generate predictions and results for BERT
def generate_bert_results():
    # Load BERT text samples
    with open(bert_samples_file, 'r', encoding='utf-8') as file:
        bert_text_samples = file.read().splitlines()
    
    # Skip the first sample
    bert_text_samples = bert_text_samples[1:] if bert_text_samples else []
    
    # Output file for BERT text predictions
    bert_output_file = r"C:\Users\kathr\OneDrive\Documents\mod_project\bert_text.txt"
    
    # Write BERT text prediction results to file
    with open(bert_output_file, 'w', encoding='utf-8') as file:
        for i, text in enumerate(bert_text_samples):
            clean_text = text.replace('[MASK]', '')
            result = bert_predict_masked_text(text, i)
            if '[MASK]' not in result:
                file.write(f"Sample {i + 1}: {clean_text}\nPrediction: {result}\n\n")
    
    print(f"BERT predictions written to {bert_output_file}")
    
    # Load QA pairs and process
    qa_pairs_file = r"C:\Users\kathr\OneDrive\Documents\mod_project\extracted_qa.json"
    try:
        with open(qa_pairs_file, 'r', encoding='utf-8') as file:
            qa_pairs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        qa_pairs = []
    
    # Output file for BERT QA results
    bert_qa_file = r"C:\Users\aghiya\OneDrive\Documents\mod_project\bert_answers.txt"
    
    # Write BERT QA results to file
    with open(bert_qa_file, 'w', encoding='utf-8') as file:
        for i, pair in enumerate(qa_pairs):
             
            result = bert_question_answering(pair["question"], pair["context"])
    
            file.write(f"Question {i + 1}: {pair['question']}\nPredicted Answer: {result}\n\n")
    
    # Generate audio for BERT text prediction
    with open(bert_output_file, 'r', encoding='utf-8') as file:
        bert_text_prediction_results = file.read()
    
    text_to_speech(f"BERT text prediction results are:\n{bert_text_prediction_results}", "bert_text.mp3")
    
    # Generate audio for BERT question answering
    with open(bert_qa_file, 'r', encoding='utf-8') as file:
        bert_qa_results = file.read()
    
    text_to_speech(f"BERT Question Answering results are:\n{bert_qa_results}", "bert_answers.mp3")
    
    print("BERT results generated and saved to files.")

# Main menu function with model selection
def main_menu():
    while True:
        print("\nMain Menu:")
        print("1. Generate and work with RoBERTa results")
        print("2. Generate and work with BERT results")
        print("3. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            print("\nGenerating RoBERTa results...")
            generate_roberta_results()
            roberta_submenu()
        elif choice == '2':
            print("\nGenerating BERT results...")
            generate_bert_results()
            bert_submenu()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

# RoBERTa submenu
def roberta_submenu():
    while True:
        print("\nRoBERTa Menu:")
        print("1. Play RoBERTa Text Prediction Results")
        print("2. Play RoBERTa Question Answering Results")
        print("3. Return to Main Menu")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            play_audio('roberta_text.mp3')
        elif choice == '2':
            play_audio('roberta_answers.mp3')
        elif choice == '3':
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please try again.")

# BERT submenu
def bert_submenu():
    while True:
        print("\nBERT Menu:")
        print("1. Play BERT Text Prediction Results")
        print("2. Play BERT Question Answering Results")
        print("3. Return to Main Menu")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            play_audio('bert_text.mp3')
        elif choice == '2':
            play_audio('bert_answers.mp3')
        elif choice == '3':
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the main menu
if __name__ == "__main__":
    main_menu()
