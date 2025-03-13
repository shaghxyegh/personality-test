from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from flask_cors import CORS
import os
import logging
import whisper
import tempfile
import time

# Configure paths
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load Whisper model
model = whisper.load_model("base")

# Load datasets and initialize responses
try:
    adult_df = pd.read_excel("Personality_Test.xlsx")
    child_df = pd.read_excel("Child_Mode_Questions.xlsx")
    responses_file = "User_Responses.xlsx"
    if os.path.exists(responses_file):
        os.remove(responses_file)
        app.logger.debug(f"Deleted existing {responses_file}")
    responses_df = pd.DataFrame(columns=['Mode', 'Question', 'Options', 'User_Answer', 'Timestamp'])
except Exception as e:
    logging.error(f"Error during initialization: {str(e)}")
    raise

# Application state
current_mode = "adult"
user_responses = []
question_index = 0


def save_response_to_excel(mode, question, options, answer):
    global responses_df
    try:
        new_response = pd.DataFrame({
            'Mode': [mode],
            'Question': [question],
            'Options': [', '.join(str(opt) for opt in options)],
            'User_Answer': [answer],
            'Timestamp': [pd.Timestamp.now()]
        })
        responses_df = pd.concat([responses_df, new_response], ignore_index=True)
        responses_df.to_excel(responses_file, index=False)
        app.logger.debug(f"Saved response to {responses_file}")
    except Exception as e:
        app.logger.error(f"Error saving to Excel: {str(e)}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode, question_index, user_responses
    try:
        data = request.get_json()
        current_mode = "child" if data.get('child_mode', False) else "adult"
        question_index = 0
        user_responses = []
        app.logger.debug(f"Mode changed to: {current_mode}")
        return jsonify({"status": "success", "mode": current_mode})
    except Exception as e:
        app.logger.error(f"Mode change error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_question', methods=['GET'])
def get_question():
    global question_index
    try:
        df = child_df if current_mode == "child" else adult_df

        if question_index >= len(df):
            return jsonify({"status": "completed"})

        question = df.iloc[question_index]['Question']

        if current_mode == "child":
            image_filenames = df.iloc[question_index, 1:5].dropna().tolist()
            image_paths = [f"/images/{filename.strip()}" for filename in image_filenames]
            response = {
                "question": question,
                "options": image_paths,
                "voice_options": [],  # Mute options in child mode
                "mode": current_mode,
                "index": question_index
            }
        else:
            options = df.iloc[question_index, 1:5].dropna().tolist()
            response = {
                "question": question,
                "options": options,
                "voice_options": options,  # Include options for voice in adult mode
                "mode": current_mode,
                "index": question_index
            }

        app.logger.debug(f"Serving question {question_index}: {question}")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Get question error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global question_index
    try:
        data = request.get_json()
        answer = data.get('answer')
        df = child_df if current_mode == "child" else adult_df

        question = df.iloc[question_index]['Question']
        options = df.iloc[question_index, 1:5].tolist()

        user_responses.append(answer)
        save_response_to_excel(current_mode, question, options, answer)

        question_index += 1
        return jsonify({"status": "success", "next": question_index < len(df)})
    except Exception as e:
        app.logger.error(f"Submit answer error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['file']
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        temp_path = temp_file.name
        audio_file.save(temp_path)
        temp_file.close()

        result = model.transcribe(temp_path)

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                os.unlink(temp_path)
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(0.1)
                    continue
                app.logger.warning(f"Could not delete temporary file: {temp_path}")

        app.logger.debug(f"Transcription result: {result['text']}")
        return jsonify({"text": result['text']})
    except Exception as e:
        app.logger.error(f"Transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/images/<path:filename>')
def serve_static(filename):
    return send_from_directory('images', filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)