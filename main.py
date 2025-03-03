from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from colloring import DeepLabModel, label_names, download_and_load_model
from PIL import Image
import numpy as np

MODEL = download_and_load_model()

app = Flask(__name__)

UPLOAD_FOLDER = r'uploaded_images'
NUMPY_FOLDER = r'uploaded_images\numpy_files'
RESIZED_FOLDER = r'uploaded_images\resized_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(NUMPY_FOLDER):
    os.makedirs(NUMPY_FOLDER)
if not os.path.exists(RESIZED_FOLDER):
    os.makedirs(RESIZED_FOLDER)


@app.route('/imagetolabel', methods=['POST'])
def imagetolabel():
    try:
        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        
        # Process the image
        original_im = Image.open(image_path)
        resized_im, seg_map = MODEL.run(original_im)
        
        # Get unique labels
        unique_labels = np.unique(seg_map)
        
        # Debug: Check if labels exist in label_names
        for label in unique_labels:
            if label in label_names.values():
                print(f"Label {label} found in label_names")
            else:
                print(f"Label {label} NOT found in label_names")
        
        detected_objects = {label: key for key, label in label_names.items() if label in unique_labels}
        
        
        # Save the processed data for the next GET request
        seg_map_path = os.path.join(NUMPY_FOLDER, f'seg_map.npy')
        np.save(seg_map_path, seg_map)
        resized_im_path = os.path.join(RESIZED_FOLDER, f'resized_im.png')
        resized_im.save(resized_im_path)
        os.remove(image_path)

        
        return jsonify({"detected_objects": detected_objects})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_image', methods=['GET'])
def process_image():
    try:
        # Get the RGB values and key from the request arguments
        r = int(request.args.get('r'))
        g = int(request.args.get('g'))
        b = int(request.args.get('b'))
        key = request.args.get('key')
    
        seg_map_path = os.path.join(NUMPY_FOLDER, f'seg_map.npy')
        resized_im_path = os.path.join(RESIZED_FOLDER, f'resized_im.png')
        
        # Debug: Print the paths
        print(f"seg_map_path: {seg_map_path}")
        print(f"resized_im_path: {resized_im_path}")
        
        if not seg_map_path or not resized_im_path:
            return jsonify({"error": "Missing seg_map_path or resized_im_path"}), 400
        
        if key not in label_names:
            return jsonify({"error": "Invalid key provided"}), 400
        
        value = label_names[key]
        print(value)
        
        # Load the saved data
        seg_map = np.load(seg_map_path)
        resized_im = Image.open(resized_im_path)
        
        w, h = resized_im.size
        for i in range(h):
            for j in range(w):
                if seg_map[i][j] == value:
                    seg_map[i][j] = 1
                else:
                    seg_map[i][j] = 0
        
        mask_j = np.repeat(seg_map[..., None], 3, axis=2)
        mask_j[np.where((mask_j == [1, 1, 1]).any(axis=2))] = [r, g, b]
        
        output_image = np.array(resized_im) + mask_j
        output_image = Image.fromarray(output_image.astype('uint8'))
        
        output_image_path = os.path.join(UPLOAD_FOLDER, 'output_image.png')
        output_image.save(output_image_path)
        
        
        return send_file(output_image_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()