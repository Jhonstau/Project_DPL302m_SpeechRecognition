from utils.architecture import Conv3D
from utils.audio.processor import audio_processor
import utils.file_manager as file_manager
import utils.model as model
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def test_audio():

    if not request.files or 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    temp_files = []

    try:
        for key in request.files:
            audio_file = request.files[key]
            
            if not audio_file.filename.lower().endswith('.wav'):
                return jsonify({"error": f"File {audio_file.filename} is not a WAV file"}), 400
            
            audio_data = audio_file.read()
            if len(audio_data) > 10 * 1024 * 1024:
                return jsonify({"error": f"File {audio_file.filename} is too large"}), 400
            
            temp_file = f"temp_{key}_{os.urandom(8).hex()}.wav"
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            temp_files.append(temp_file)
            
            processor = audio_processor()
            data = processor.batch_preprocess(temp_files)

            pretrained_model = model.load("models/3DCNN_modelV2_fixed.pt")
            results = model.predict(data, pretrained_model)

        return jsonify({"emotion": results[0][0]}), 200

    except Exception as e:
        return jsonify({"error": f"{str(e)}"}), 500

    finally:
        file_manager.clean_up(temp_files)

if __name__ == '__main__':
    app.run(debug=True)