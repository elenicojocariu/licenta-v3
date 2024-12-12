from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from extruder import detect_edges, generate_depth_map
import warnings
from depth_processor import process_all_depth_maps

warnings.simplefilter('default')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/extrude', methods=['POST'])
def extrude():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # verif sau gen harta de  h
    depth_map_path = generate_depth_map(filename, source_folder=UPLOAD_FOLDER)
    print(f"Harta de adâncime folosită: {depth_map_path}")

    # Detectarea contururilor (opțional)
    output_path = detect_edges(file_path)
    print(f"Imaginea procesată cu contururi: {output_path}")

    #postprocessed = overlay_edges_on_original(file_path)
    #print(f"img postprocesata: {postprocessed}")

    return send_file(output_path, as_attachment=True)


@app.route('/depth_exists', methods=['POST'])
def depth_exists():
    data = request.json
    image_name = data.get('image_name')

    # Calea imaginii din depth
    depth_path = os.path.join("depth_maps", image_name)

    if os.path.exists(depth_path):  # Verifică dacă imaginea din depth există
        return jsonify({"exists": True, "processed_image_path": f"/processed/edges_{image_name}"})

    # Chiar dacă imaginea din depth nu există, returnăm calea processed
    return jsonify({"exists": False, "processed_image_path": f"/processed/edges_{image_name}"})


if __name__ == "__main__":
    depth_maps_directory = "depth_maps"
    process_all_depth_maps(depth_maps_directory)
    app.run(port=5001)
