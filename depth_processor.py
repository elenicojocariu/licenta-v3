import os
from extruder import create_3d_mesh_with_texture
UPLOAD_FOLDER = "uploads"


def process_all_depth_maps(depth_maps_dir):
    """
   Procesează toate hărțile de adâncime și generează modele 3D cu textură.
   """


    if not os.path.exists(depth_maps_dir):
        print(f"Directorul {depth_maps_dir} nu există!")
        return

    for file_name in os.listdir(depth_maps_dir):
        depth_map_path = os.path.join(depth_maps_dir, file_name)

        # Verificăm dacă fișierul este o hartă de adâncime validă
        if not file_name.endswith(("_depth.jpg", "_depth.png")):
            print(f"Fișierul {file_name} nu este o hartă de adâncime validă. Se trece peste.")
            continue

        # Eliminăm sufixul "_depth" pentru a găsi imaginea originală
        original_image_name = file_name.replace("_depth", "")
        original_image_path = os.path.join(UPLOAD_FOLDER, original_image_name)

        # Verificăm dacă imaginea originală există
        if not os.path.exists(original_image_path):
            print(f"Fișierul original {original_image_path} nu există. Se trece peste.")
            continue

        print(f"Procesăm fișierul: {depth_map_path} cu imaginea originală: {original_image_path}")
        try:
            create_3d_mesh_with_texture(original_image_path, depth_map_path, z_scale=2.0)
        except Exception as e:
            print(f"Eroare la procesarea fișierului {depth_map_path}: {e}")
