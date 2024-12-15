from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import subprocess

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def homenav():
    return render_template('home.html')

# @app.route('/audiotospectrum')
# def atosp():
#     return render_template('/audiotospectrum.html')


# @app.route('/spectrumtoaudio')
# def spectoa():
#     return render_template('/spectrumtoaudio.html')


@app.route('/record', methods=['POST'])
def record_audio():
    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(audio_path)

    # Convert to spectrogram
    spectrogram_filename = Path(filename).stem + '.png'
    spectrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], spectrogram_filename)
    
    try:
        result = subprocess.run(
            ['python', 'audio_to_spectrogram.py', '--source', audio_path, '--output', spectrogram_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error generating spectrogram: {e.stderr}")
        return f"Error generating spectrogram: {e.stderr}"

    # Return filename for spectrogram
    return render_template('index.html', spectrogram_file=spectrogram_filename)

@app.route('/upload', methods=['POST'])
def upload_spectrogram():
    spectrogram_file = request.files['file']
    filename = secure_filename(spectrogram_file.filename)
    spectrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    spectrogram_file.save(spectrogram_path)

    # Convert to audio
    audio_filename = Path(filename).stem + '.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    try:
        result = subprocess.run(
            ['python', 'spectrogram_to_audio.py', '--spectrogram', spectrogram_path, '--output', audio_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error generating audio: {e.stderr}")
        return f"Error generating audio: {e.stderr}"

    # Return filename for audio
    return render_template('index.html', audio_file=audio_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
