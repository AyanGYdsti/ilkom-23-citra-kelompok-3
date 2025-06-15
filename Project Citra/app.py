from flask import Flask, render_template, request, send_from_directory, redirect, url_for, send_file
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
from rembg import remove
from fpdf import FPDF
import shutil


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ENHANCED_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced')
app.config['PDF_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'pdf')
app.config['BOOSTED_FOLDER'] = 'static/boosted'
app.config['THRESHOLD_FOLDER'] = 'static/thresholded'

UPLOAD_FOLDER = "uploads"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)
os.makedirs(app.config['BOOSTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['THRESHOLD_FOLDER'], exist_ok=True)

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
    original_image_data = None
    original_snippet = None
    enhanced_snippet = None
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

        # Simpan gambar asli ke base64
        _, buffer_orig = cv2.imencode('.jpg', img)
        original_image_data = base64.b64encode(buffer_orig).decode('utf-8')

        # Enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(img_clahe, -1, kernel)

        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.multiply(hsv[..., 1], 1.1)
        hsv[..., 2] = cv2.multiply(hsv[..., 2], 1.05)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(save_path, enhanced)

        # Simpan hasil ke base64
        _, buffer = cv2.imencode('.jpg', enhanced)
        image_data = base64.b64encode(buffer).decode('utf-8')
        download_url = f"/download/{filename}"

        # Cuplikan matriks
        original_snippet = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:5, :5].tolist()
        enhanced_snippet = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)[:5, :5].tolist()

    return render_template(
        'enhance.html',
        image_data=image_data,
        original_image_data=original_image_data,
        download_url=download_url,
        original_snippet=original_snippet,
        enhanced_snippet=enhanced_snippet
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename, as_attachment=True)

@app.route('/remove-background', methods=['GET', 'POST'])
def remove_background():
    image_path = None
    original_image_path = None
    download_filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return 'No file selected'

        filename = secure_filename(file.filename)
        upload_dir = 'uploads/bg_removed'
        os.makedirs(upload_dir, exist_ok=True)

        input_path = os.path.join(upload_dir, filename)
        file.save(input_path)

        # Simpan path asli untuk preview
        original_image_path = f"/{input_path}"

        # Hapus background
        input_bytes = open(input_path, "rb").read()
        output_bytes = remove(input_bytes)

        output_filename = f"no_bg_{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(upload_dir, output_filename)
        with open(output_path, "wb") as out_file:
            out_file.write(output_bytes)

        image_path = f"/{output_path}"
        download_filename = output_filename

    return render_template(
        "remove_background.html",
        image_path=image_path,
        original_image_path=original_image_path,
        download_filename=download_filename
    )


@app.route('/uploads/bg_removed/<filename>')
def serve_removed_file(filename):
    return send_from_directory('uploads/bg_removed', filename)

@app.route('/convert-to-pdf', methods=['GET', 'POST'])
def convert_to_pdf():
    if request.method == 'POST':
        files = request.files.getlist('images')
        image_list = []

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Konversi gambar ke mode RGB
                img = Image.open(filepath).convert('RGB')
                image_list.append(img)

        if not image_list:
            return "Tidak ada gambar valid yang dipilih.", 400

        # Simpan semua gambar ke satu file PDF
        output_pdf_path = os.path.join(app.config['PDF_FOLDER'], 'output.pdf')
        image_list[0].save(output_pdf_path, save_all=True, append_images=image_list[1:])

        return redirect(url_for('download_pdf'))

    return render_template('convert_pdf.html')

@app.route('/invert', methods=['GET', 'POST'])
def invert():
    image_matrix = None
    img_url = None
    img_url_original = None  # Tambahan

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Simpan juga versi asli (berwarna) ke folder static
            output_original = os.path.join('static', 'inverted', f"original_{filename}")
            os.makedirs(os.path.dirname(output_original), exist_ok=True)
            shutil.copy(save_path, output_original)
            img_url_original = f"/static/inverted/original_{filename}"

            # Proses grayscale dan invert
            image = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)

            # Simpan sebagian matriks asli (10x10)
            image_matrix_original = image[:10, :10].tolist()

            # Inversi citra (negatif)
            inverted_image = 255 - image
            
            inverted_image = 255 - image

            image_matrix_inverted = inverted_image[:10, :10].tolist()
            
            # Simpan hasil invert
            output_filename = f"inverted_{filename}"
            output_path = os.path.join('static', 'inverted', output_filename)
            cv2.imwrite(output_path, inverted_image)

            image_matrix = inverted_image.tolist()
            img_url = f"/static/inverted/{output_filename}"

            return render_template(
                'invert.html',
                image_matrix=image_matrix,
                img_path=img_url,
                img_original_path=img_url_original,
                image_matrix_original=image_matrix_original,
                image_matrix_inverted=image_matrix_inverted
            )

    return render_template('invert.html')

@app.route('/color_boost', methods=['GET', 'POST'])
def color_boost():
    original_image_data = None
    boosted_image_data = None
    original_snippet = None
    boosted_snippet = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return 'No file selected'

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Baca gambar asli
        img = cv2.imread(save_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (600, 600))

        # Simpan gambar asli ke base64 untuk ditampilkan di HTML
        _, buffer_orig = cv2.imencode('.jpg', img)
        original_image_data = base64.b64encode(buffer_orig).decode('utf-8')

        # Konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Boost saturasi channel S
        hsv[..., 1] = hsv[..., 1] * 1.5
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

        # Konversi kembali ke BGR
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Simpan gambar boosted ke file agar bisa diakses (optional)
        boosted_filename = f"boosted_{filename}"
        boosted_path = os.path.join(app.config['BOOSTED_FOLDER'], boosted_filename)
        cv2.imwrite(boosted_path, boosted)

        # Simpan gambar boosted ke base64
        _, buffer_boosted = cv2.imencode('.jpg', boosted)
        boosted_image_data = base64.b64encode(buffer_boosted).decode('utf-8')

        # Cuplikan matriks grayscale 5x5
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boosted_gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
        original_snippet = original_gray[:5, :5].tolist()
        boosted_snippet = boosted_gray[:5, :5].tolist()

    return render_template('color_boost.html',
                            original_image_data=original_image_data,
                            boosted_image_data=boosted_image_data,
                            original_snippet=original_snippet,
                            boosted_snippet=boosted_snippet)

@app.route('/threshold', methods=['GET', 'POST'])
def threshold_image():
    original_image_data = None
    thresholded_image_data = None
    original_snippet = None
    thresholded_snippet = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return 'No file selected'

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Baca gambar dan resize
        img = cv2.imread(save_path)
        img = cv2.resize(img, (600, 600))

        # Simpan gambar asli (BGR) ke base64
        _, buffer_orig = cv2.imencode('.jpg', img)
        original_image_data = base64.b64encode(buffer_orig).decode('utf-8')

        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Simpan snippet grayscale asli
        original_snippet = gray[:5, :5].tolist()

        # Thresholding (binarisasi)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Simpan snippet hasil threshold
        thresholded_snippet = thresh[:5, :5].tolist()

        # Simpan hasil threshold ke base64
        _, buffer_thresh = cv2.imencode('.jpg', thresh)
        thresholded_image_data = base64.b64encode(buffer_thresh).decode('utf-8')

    return render_template('threshold.html',
                            original_image_data=original_image_data,
                            thresholded_image_data=thresholded_image_data,
                            original_snippet=original_snippet,
                            thresholded_snippet=thresholded_snippet)

@app.route('/brightness', methods=['GET', 'POST'])
def brightness():
    original_image_data = None
    bright_image_data = None
    original_matrix = None
    bright_matrix = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return 'No file selected'

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        img = cv2.imread(file_path)
        img = cv2.resize(img, (600, 600))

        # Simpan gambar asli (base64)
        _, buffer_orig = cv2.imencode('.jpg', img)
        original_image_data = base64.b64encode(buffer_orig).decode('utf-8')

        # Tingkatkan brightness
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)  # Brightness via beta

        # Simpan hasil brightness (base64)
        _, buffer_bright = cv2.imencode('.jpg', bright)
        bright_image_data = base64.b64encode(buffer_bright).decode('utf-8')

        # Ambil 5x5 piksel tengah (grayscale)
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bright = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)

        h, w = gray_orig.shape
        cx, cy = w // 2, h // 2
        original_matrix = gray_orig[cy - 2:cy + 3, cx - 2:cx + 3].tolist()
        bright_matrix = gray_bright[cy - 2:cy + 3, cx - 2:cx + 3].tolist()

    return render_template(
        'brightness.html',
        original_image_data=original_image_data,
        bright_image_data=bright_image_data,
        original_matrix=original_matrix,
        bright_matrix=bright_matrix
    )

@app.route('/download-pdf')
def download_pdf():
    pdf_path = os.path.join(app.config['PDF_FOLDER'], 'output.pdf')

    if not os.path.exists(pdf_path):
        return "PDF tidak ditemukan", 404

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name='output.pdf',
        mimetype='application/pdf'
    )

@app.route('/features')
def features():
    return render_template('features.html')

if __name__ == '__main__':
    app.run(debug=True)
