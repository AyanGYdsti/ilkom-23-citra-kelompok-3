from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ENHANCED_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced')

UPLOAD_FOLDER = "uploads"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['GET', 'POST'])
def index_ocr():
    extracted_text = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return 'No file selected'

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Proses berdasarkan jenis file
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(img)

        elif file.filename.lower().endswith('.pdf'):
            images = convert_from_path(filepath, poppler_path=r"C:\Poppler\Library\bin")
            text_list = [pytesseract.image_to_string(img) for img in images]
            extracted_text = "\n\n".join(text_list)

    return render_template('index_ocr.html', extracted_text=extracted_text)

@app.route('/enhance', methods=['GET', 'POST'])
def enhance_image():
    image_data = None
    download_url = None
    ENHANCED_FOLDER = app.config['ENHANCED_FOLDER']

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return 'No file selected'

        original_filename = secure_filename(file.filename)
        filename = f"enhanced_{original_filename}"
        save_path = os.path.join(ENHANCED_FOLDER, filename)

        # Baca dan proses gambar
        in_memory_file = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (600, 600))

        # Peningkatan kualitas gambar
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        sharpened = cv2.filter2D(img_clahe, -1, kernel)

        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.multiply(hsv[..., 1], 1.1)
        hsv[..., 2] = cv2.multiply(hsv[..., 2], 1.05)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Simpan hasil
        cv2.imwrite(save_path, enhanced)

        # Encode base64 untuk preview
        _, buffer = cv2.imencode('.jpg', enhanced)
        image_data = base64.b64encode(buffer).decode('utf-8')
        download_url = f"/download/{filename}"

    return render_template('enhance.html', image_data=image_data, download_url=download_url)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
