import os
from extruder import create_3d_mesh_with_texture

UPLOAD_FOLDER = "uploads"


def process_all_depth_maps(depth_maps_dir):

    if not os.path.exists(depth_maps_dir):
        print(f"Directory {depth_maps_dir} does not exist")
        return

    for file_name in os.listdir(depth_maps_dir):
        depth_map_path = os.path.join(depth_maps_dir, file_name)

        if not file_name.endswith(("_depth.jpg", "_depth.png")):
            print(f"This file {file_name} is not a valid depth map.")
            continue

        # elimin sufix pt a ajunge la img originala
        original_image_name = file_name.replace("_depth", "")
        original_image_path = os.path.join(UPLOAD_FOLDER, original_image_name)

        if not os.path.exists(original_image_path):
            print(f"This file {original_image_path} does not exist")
            continue

        print(f"Processing file {depth_map_path} with original image {original_image_path}")
        try:
            create_3d_mesh_with_texture(original_image_path, depth_map_path, z_scale=1.5)
        except Exception as e:
            print(f"Error at processing {depth_map_path}: {e}")
