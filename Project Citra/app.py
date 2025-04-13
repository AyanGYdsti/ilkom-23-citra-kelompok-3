from flask import Flask, render_template, request
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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


if __name__ == '__main__':
    app.run(debug=True)
