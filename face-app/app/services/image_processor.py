import os
from PIL import Image
import requests

def process_image(image_file):
    try:
        input_path = os.path.join('app/static', image_file.filename)
        image_file.save(input_path)

        with open(input_path, 'rb') as f:
            response = requests.post(
                # 'http://external-application/api/process', 
                # files={'image': f}
            )

        if response.status_code != 200:
            return None
        
        # revoir le format de la réponse 
        image_path = response.image_path
       

        return image_path

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
