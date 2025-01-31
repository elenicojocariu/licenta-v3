import cv2
import os

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

import open3d as o3d

PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MESHES_FOLDER = "meshes"
os.makedirs(MESHES_FOLDER, exist_ok=True)


def generate_depth_map(image_path, source_folder="uploads"):
    H_MAPS_FOLDER = "depth_maps"
    os.makedirs(H_MAPS_FOLDER, exist_ok=True)

    if image_path.endswith(".jpg"):
        depth_map_filename = os.path.basename(image_path).replace(".jpg", "_depth.jpg")
    elif image_path.endswith(".png"):
        depth_map_filename = os.path.basename(image_path).replace(".png", "_depth.png")
    else:
        raise ValueError("Unsupported type format. Only .jpg and .png extensions are accepted.")

    depth_map_path = os.path.join(H_MAPS_FOLDER, depth_map_filename)
    if os.path.exists(depth_map_path):
        print(f"Depth map already exists: {depth_map_path}")
        return depth_map_path

    transform_function = Compose([
        Resize(384),  # 384 inaltime si latimea redimensionata automat
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))  # tuplu cu un sg element
        # formula pentru standardizare e pixel_normalizat = (pixel_original - media) / std
    ])

    model_type = "DPT_Large"
    model_imported_from_github = torch.hub.load("intel-isl/MiDaS", model_type)
    model_imported_from_github.eval()  # switch on la predictii

    final_image_path = os.path.join(source_folder, os.path.basename(image_path))

    img_orig = Image.open(final_image_path).convert("RGB")
    image_tensor = transform_function(img_orig).unsqueeze(0)  # pe pozitia0

    # start generare harta de adancime
    with torch.no_grad():
        prediction = model_imported_from_github(image_tensor)  # predictia propriu zisa
        depth_map = prediction.squeeze().cpu().numpy()

    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(depth_map_path, depth_map_normalized)
    print(f"Depth map generated and saved: {depth_map_path}")

    return depth_map_path


def convert_to_gltf(obj_path, gltf_path):
    if not os.path.exists(obj_path):
        raise ValueError(f"OBJ file does not exist: {obj_path}")

    # citesc din obj
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_triangles():
        raise ValueError("The mesh does not contain valid triangles.")

    # scriu in gltf
    o3d.io.write_triangle_mesh(gltf_path, mesh, write_triangle_uvs=True)
    print(f"Mesh saved and converted to GLTF: {gltf_path}")


GLTF_FOLDER = "gltf_meshes"
os.makedirs(GLTF_FOLDER, exist_ok=True)


def create_3d_mesh_with_texture(image_path, depth_map_path, z_scale=1.5):
    original_image = cv2.imread(image_path)  # BGR default

    original_image_RGB = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Couldn't load depth map: {depth_map_path}")

    # Normalizez adancimea
    depth_map = depth_map.astype(np.float32) / 255.0
    depth_map *= z_scale * 50.0

    depth_map_height, depth_map_width = depth_map.shape

    # ca sa corespunda cu meshul 3d
    original_image_resized = cv2.resize(original_image_RGB, (depth_map_width, depth_map_height))
    print(f"Original image redimensioned at: {depth_map_width} x {depth_map_height}")

    # puncte din imagine pregatite pt mesh si culorile punctelor
    mesh_vertices = []
    colors = []
    for y in range(depth_map_height):
        for x in range(depth_map_width):
            z = depth_map[y, x]
            color = original_image_resized[y, x] / 255.0
            mesh_vertices.append((x, depth_map_height - y - 1, z))
            colors.append(color)

    mesh_vertices = np.array(mesh_vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    triangles = []
    for y in range(depth_map_height - 1):
        for x in range(depth_map_width - 1):
            v0 = y * depth_map_width + x  # linia*width+coloana
            v1 = v0 + 1
            v2 = v0 + depth_map_width  # cel de deasupra lui+ cati mai sunt pana la el
            v3 = v2 + 1
            triangles.append((v0, v2, v1))
            triangles.append((v1, v2, v3))

    triangles = np.array(triangles, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    # print(f"mesh: {mesh}") 0 points and 0 triangles
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    original_image_height, original_image_width, _ = original_image.shape

    # print(f"Dimensiuni imag originala: {width} x {height} (width x height)")
    x_coords = mesh_vertices[:, 0]  # vertices[row, cloumn]
    y_coords = mesh_vertices[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    print(f"Dimensions of 3D extruded model: width: {x_max - x_min:.2f}, height: {y_max - y_min:.2f}")

    # verif fata triunghiurilor si le reorintam corect
    mesh.compute_triangle_normals()
    mesh.orient_triangles()

    base_name = os.path.splitext(os.path.basename(image_path))[0]  # painting-11

    obj_path = os.path.join(MESHES_FOLDER, f"{base_name}_extruded.obj")
    # print(f"objjjjjjjjjj path {obj_path}") meshes\painting-11_extruded.obj
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    print(f"3D Mesh saved in: {obj_path}")

    # vizualizare mesh
    # o3d.visualization.draw_geometries([mesh])

    # OBJ -> GLTF
    gltf_path = os.path.join(GLTF_FOLDER, f"{base_name}_extruded.gltf")
    convert_to_gltf(obj_path, gltf_path)

    return obj_path
