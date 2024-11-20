import os
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS

from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"webm", "wav", "mp3"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL = whisper.load_model("tiny")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

 
def transcribe_file(audio_file: str):
    result = MODEL.transcribe(audio_file)
    return result["text"]

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file in request"}), 400

        file = request.files["audio"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.webm"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            transcribed_text = transcribe_file(file_path)
            return jsonify({"success": True, "transcribed_text": transcribed_text}), 200
        except Exception as e:
            print("Exception: ", e)
            return jsonify({"error", "error transcribing"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
