from app import create_app
from flask_cors import CORS
from flask_socketio import SocketIO
from face_analyser import main
import os
import base64

app = create_app()
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3030")

@socketio.on('process_image')
def handle_image_request(json_data):
    """
    Handle image processing via WebSocket.
    """
    image_path = json_data.get('path')
    process_result = main(image_path)

    if not image_path or not os.path.exists(image_path):
        socketio.emit('response', {"error": "Invalid image path"})
        return
    try:
        with open("../data/output/best_match.jpg", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Envoyer l'image encodée en base64
        socketio.emit('response', {
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "result_path": process_result["path"], 
            "score": process_result["score"]
        })

    except Exception as e:
        socketio.emit('response', {"error": str(e)})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5050, allow_unsafe_werkzeug=True)
