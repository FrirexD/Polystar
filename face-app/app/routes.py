from flask import Blueprint, request, jsonify
from flask_socketio import SocketIO
from app.services.image_processor import process_image
import os
import requests
from datetime import datetime
from werkzeug.utils import secure_filename

bp = Blueprint('routes', __name__)
UPLOAD_FOLDER = "./data/samples"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Vérifie si l'extension du fichier est valide."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Génère un nom de fichier unique avec timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    secure_name = secure_filename(original_filename)
    filename, extension = os.path.splitext(secure_name)
    return f"{filename}_{timestamp}{extension}"

@bp.route('/process-image', methods=['POST'])
def process_image_route():
    # Vérifie si un fichier a été envoyé
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    # Vérifie si le fichier a un nom et une extension autorisée
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type. Only .png, .jpg, and .jpeg are allowed.'}), 400

    # Appelle le service pour traiter l'image
    processed_image_path = process_image(image_file)

    if not processed_image_path:
        return jsonify({'error': 'Error processing the image'}), 500

    return jsonify({'result': 'Image processed successfully', 'path': processed_image_path, 'score': 100})

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        # Créer un nom de fichier unique avec timestamp
        unique_filename = generate_unique_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        try:
            file.save(filepath)
            return jsonify({
                'success': True,
                'path': filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file'}), 400