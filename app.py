from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import uuid
from realesrgan import RealESRGAN
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    files = request.files.getlist('images')
    output_files = []

    for file in files:
        if file and file.filename:
            filename = str(uuid.uuid4()) + '.png'
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            file.save(input_path)

            with open(input_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                sr_image = model.predict(img)
                sr_image.save(output_path)

            output_files.append(filename)

    return render_template('result.html', files=output_files)

@app.route('/outputs/<filename>')
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
