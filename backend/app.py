from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  


app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route de test
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "success",
        "message": "L'API est opérationnelle"
    })

# Route pour uploader une image
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "Aucun fichier n'a été envoyé"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "Aucun fichier sélectionné"
        }), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            "status": "success",
            "message": "Fichier uploadé avec succès",
            "filename": filename
        })
    
    return jsonify({
        "status": "error",
        "message": "Type de fichier non autorisé"
    }), 400

# Point d'entrée pour lancer l'application
if __name__ == '__main__':
    app.run(debug=True)