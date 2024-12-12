import os
from extruder import depth_map_to_3d


def process_all_depth_maps(depth_maps_dir):
    """
    Procesează toate hărțile de adâncime și generează modele 3D.
    """
    if not os.path.exists(depth_maps_dir):
        print(f"Directorul {depth_maps_dir} nu există!")
        return

    for file_name in os.listdir(depth_maps_dir):
        depth_map_path = os.path.join(depth_maps_dir, file_name)
        # Eliminăm sufixul "_depth" din numele fișierului pentru a găsi imaginea originală
        original_image_name = file_name.replace("_depth", "").replace("overlay_", "")
        original_image_path = os.path.join("uploads", original_image_name)

        # Verifică dacă fișierul de adâncime există
        if not os.path.exists(depth_map_path):
            print(f"Fișierul de adâncime {depth_map_path} nu există. Se trece peste.")
            continue

        # Verifică dacă fișierul original există
        if not os.path.exists(original_image_path):
            print(f"Fișierul original {original_image_path} nu există. Se trece peste.")
            continue

        if os.path.isfile(depth_map_path) and file_name.endswith((".jpg", ".png")):
            print(f"Procesăm harta de adâncime: {depth_map_path}")
            try:
                depth_map_to_3d(depth_map_path)
            except Exception as e:
                print(f"Eroare la procesarea fișierului {depth_map_path}: {e}")
