from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from extruder import generate_depth_map, create_3d_mesh_with_texture
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
    depth_map_filename = f"{os.path.splitext(filename)[0]}_depth.jpg"
    depth_map_path = os.path.join("depth_maps", depth_map_filename)

    if not os.path.exists(depth_map_path):
        print(f"Generating depth map for {filename}.....")
        depth_map_path = generate_depth_map(filename, source_folder=UPLOAD_FOLDER)
    else:
        print(f"Depth map already exists: {depth_map_path}")

    gltf_filename = f"{os.path.splitext(filename)[0]}_extruded.gltf"
    gltf_path = os.path.join("gltf_meshes", gltf_filename)

    if not os.path.exists(gltf_path):
        print(f"Generating GLTF mesh for {filename}.....")
        created_gltf_path = create_3d_mesh_with_texture(file_path, depth_map_path, z_scale=1.5)
    else:
        print(f"GLTF mesh already exists: {gltf_path}")

    return jsonify({
        "message": "extrusion complted",
        "depth_map_path": f"/depth_maps/{depth_map_filename}",
        "gltf_path": f"gltf_meshes/{gltf_filename}"
    }), 200


@app.route('/depth_exists', methods=['POST'])
def depth_exists():
    data = request.json

    if 'image_name' not in data:
        return "No file for depth_map uploaded", 400

    image_name = data.get('image_name')
    depth_path = os.path.join("depth_maps", image_name)

    if os.path.exists(depth_path):
        return jsonify({"exists": True, "processed_image_path": f"/processed/edges_{image_name}"})

    return jsonify({"exists": False, "processed_image_path": f"/processed/edges_{image_name}"})


@app.route('/gltf_exists', methods=['POST'])
def gltf_exists():
    data = request.json
    image_name = data.get('image_name')

    gltf_path = os.path.join("gltf_meshes", f"{os.path.splitext(image_name)[0]}_extruded.gltf")

    if os.path.exists(gltf_path):
        return jsonify({"exists": True, "gltf_path": f"/gltf_meshes/{os.path.basename(gltf_path)}"})

    return jsonify({"exists": False, "gltf_path": None})


@app.route('/send_mesh', methods=['POST'])
def send_mesh():
    data = request.json
    image_name = data.get('image_name')

    gltf_path = os.path.join("gltf_meshes", f"{os.path.splitext(image_name)[0]}_extruded.gltf")

    if os.path.exists(gltf_path):
        return send_file(gltf_path, as_attachment=True)

    return jsonify({"error": "Mesh not found"}), 404


if __name__ == "__main__":
    depth_maps_directory = "depth_maps"
    process_all_depth_maps(depth_maps_directory)

    app.run(port=5001)
