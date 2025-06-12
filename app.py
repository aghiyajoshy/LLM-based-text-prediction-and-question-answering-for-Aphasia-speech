from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import platform
from werkzeug.utils import secure_filename
from combined import (
    process_transcript_for_roberta,
    process_transcript_for_bert,
    roberta_predict_masked_text,
    bert_predict_masked_text,
    roberta_question_answering,
    bert_question_answering,
    aai
)

# ==================== SETUP ====================
app = Flask(__name__)
CORS(app)  # Allow all origins

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_to_speech(text, output_file, play_audio=True):
    """Convert text to speech and optionally play the audio"""
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    engine.save_to_file(text, output_file)
    engine.runAndWait()
    print(f"Speech saved to '{output_file}'")

    if play_audio:
        if platform.system() == "Windows":
            os.system(f'start {output_file}')
        elif platform.system() == "Darwin":  # macOS
            os.system(f"afplay '{output_file}'")
        else:  # Linux
            os.system(f"mpg123 '{output_file}'")

def extract_qa_pairs(transcript_text):
    """Extract question-answer pairs from transcript text (placeholder)"""
    qa_pairs = [
        {"question": "How are you feeling today?", "context": transcript_text},
        {"question": "Can you tell me about your day?", "context": transcript_text},
        {"question": "Do you need any help with anything?", "context": transcript_text},
        {"question": "Is there anything specific you'd like to discuss?", "context": transcript_text},
    ]
    return qa_pairs

def run_speech_analysis_pipeline(audio_file_path, model_choice='bert', play_audio=False):
    """Run the full analysis pipeline on the uploaded audio file"""
    # Set AssemblyAI API key
    aai.settings.api_key = "56481a160f944ad88be6220e68114dea"

    # Initialize transcriber with disfluencies enabled
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        disfluencies=True,
        speaker_labels=True
    )
    
    # Transcribe audio
    transcript = transcriber.transcribe(audio_file_path, config=config)
    transcript_text = transcript.text
    
    if model_choice.lower() == 'roberta':
        masked_sentences = process_transcript_for_roberta(transcript_text)
        output_file = os.path.join(UPLOAD_FOLDER, "roberta_text_prediction_results.txt")

        with open(output_file, 'w', encoding='utf-8') as file:
            for i, text in enumerate(masked_sentences, start=1):
                if '<mask>' not in text and '[MASK]' not in text:
                    file.write(f"Sample {i}:\nInput: {text}\nNo mask token found, skipping.\n\n")
                    continue
                result = roberta_predict_masked_text(text)
                result = result.replace('[MASK]', '').replace('<mask>', '').strip()
                file.write(f"Sample {i}:\nInput: {text}\nPrediction: {result}\n\n")

        qa_pairs = extract_qa_pairs(transcript_text)
        qa_output_file = os.path.join(UPLOAD_FOLDER, "roberta_question_answering_results.txt")

        with open(qa_output_file, 'w', encoding='utf-8') as file:
            for i, pair in enumerate(qa_pairs):
                result = roberta_question_answering(pair["question"], pair["context"])
                file.write(f"Question {i + 1}: {pair['question']}\nPredicted Answer: {result}\n\n")

        if play_audio:
            text_to_speech(read_file_content(output_file), os.path.join(UPLOAD_FOLDER, "roberta_text.mp3"))
            text_to_speech(read_file_content(qa_output_file), os.path.join(UPLOAD_FOLDER, "roberta_answers.mp3"))

    else:
        masked_sentences = process_transcript_for_bert(transcript_text)
        output_file = os.path.join(UPLOAD_FOLDER, "bert_text_prediction_results.txt")

        with open(output_file, 'w', encoding='utf-8') as file:
            for i, text in enumerate(masked_sentences):
                clean_text = text.replace('[MASK]', '')
                result = bert_predict_masked_text(text, i)
                file.write(f"Sample {i + 1}: {clean_text}\nPrediction: {result}\n\n")

        qa_pairs = extract_qa_pairs(transcript_text)
        qa_output_file = os.path.join(UPLOAD_FOLDER, "bert_question_answering_results.txt")

        with open(qa_output_file, 'w', encoding='utf-8') as file:
            for i, pair in enumerate(qa_pairs):
                result = bert_question_answering(pair["question"], pair["context"])
                file.write(f"Question {i + 1}: {pair['question']}\nPredicted Answer: {result}\n\n")

        if play_audio:
            text_to_speech(read_file_content(output_file), os.path.join(UPLOAD_FOLDER, "bert_text.mp3"))
            text_to_speech(read_file_content(qa_output_file), os.path.join(UPLOAD_FOLDER, "bert_answers.mp3"))

def read_file_content(file_path):
    """Read text content from a file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# ==================== ROUTES ====================
@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    model_choice = request.form.get('model', 'bert')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            run_speech_analysis_pipeline(filepath, model_choice=model_choice, play_audio=True)
            return jsonify({'status': 'Analysis complete', 'file': filename}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# ==================== RUN APP ====================
if __name__ == '__main__':
    app.run(debug=True)
