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


'''def generate_3d_mesh(depth_map, edge_map=None):
    """
    Convertește o hartă de adâncime într-un model 3D extrudat.
    """
    height, width = depth_map.shape

    # Generează punctele 3D din harta de adâncime
    points = []
    colors = []

    depth_scale = 100

    for y in range(height):
        for x in range(width):
            z = depth_map[y, x] * depth_scale
            if z > 0:
                points.append([x, height - y, z])  # Inversăm coordonata Y
                colors.append([z / depth_scale, z / depth_scale, z / depth_scale])  # Scală de gri pentru culori
    points = np.array(points)
    colors = np.array(colors)

    # Adăugăm punctele marginilor pentru detalii
    if edge_map is not None:
        edge_points = np.argwhere(edge_map > 0)
        for y, x in edge_points:
            points = np.append(points, [[x, y, depth_map[y, x]]], axis=0)
            colors = np.append(colors, [[1, 0, 0]], axis=0)
        print(f"Numar de puncte: {len(points)}, Culori adăugate: {len(colors)}")

    # Creează un nor de puncte
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Reconstrucție mesh 3D folosind Poisson Surface Reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    print(f"Dimensiuni model extrdat: width: {width: .2f}, height: {height: .2f}")
    return mesh
'''

'''def depth_map_to_3d(depth_map_path):
    """
    Procesează o hartă de adâncime pentru a genera un model 3D.
    """
    depth_map = preprocess_depth_map(depth_map_path)
    print(f"Dimensiuni hartă de adâncime: {depth_map.shape[1]} x {depth_map.shape[0]} (W x H)")

    base_name = os.path.basename(depth_map_path).replace(".jpg", "")
    edge_map_path = os.path.join(PROCESSED_FOLDER, f"edges_{base_name}.jpg")

    edge_map = None
    if os.path.exists(edge_map_path):
        edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)

    mesh = generate_3d_mesh(depth_map, edge_map)

    # Load the original image as a texture
    base_name = os.path.basename(depth_map_path).replace("_depth", "").replace(".jpg", "")
    original_image_path = os.path.join("uploads", f"{base_name}.jpg")

    if not os.path.exists(original_image_path):
        print(f"Texture image {original_image_path} not found.")
        return

    texture_image = cv2.imread(original_image_path)
    if texture_image is None:
        print(f"Failed to read texture image: {original_image_path}")
        return
    # print(f"Dimensiuni imagine originală (textură): {texture_image.shape[1]} x {texture_image.shape[0]} (W x H)")

    # Ensure the texture image has the same dimensions as the depth map
    if (texture_image.shape[1], texture_image.shape[0]) != (depth_map.shape[1], depth_map.shape[0]):
        texture_image = cv2.resize(texture_image, (depth_map.shape[1], depth_map.shape[0]))
        print(f"Dimensiune imagine originala dupa resize: {texture_image.shape[1]} x {texture_image.shape[0]}")
    # Convert texture to RGB format for Open3D
    texture_image = np.asarray(texture_image, dtype=np.float32) / 255.0  # Textura redimensionată ca float32

    texture_pil = Image.fromarray((texture_image * 255).astype(np.uint8))
    texture_path = os.path.join(POSTPROCESSED_FOLDER, f"texture_{base_name}.png")
    texture_pil.save(texture_path)  # Save the texture image in PNG format for Open3D

    # Încarcă textura din fișierul salvat
    texture_pil = Image.open(texture_path)
    texture_np = np.asarray(texture_pil, dtype=np.uint8)
    texture_o3d = o3d.geometry.Image(texture_np)

    # Assign UV coordinates and set the texture to the mesh
    mesh.compute_vertex_normals()
    uv_map = np.zeros((np.asarray(mesh.vertices).shape[0], 2))

    # Simple UV mapping: normalized (x, y) positions
    vertices = np.asarray(mesh.vertices)  # Convert to NumPy array
    uv_map[:, 0] = (vertices[:, 0] - np.min(vertices[:, 0])) / (np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    uv_map[:, 1] = (vertices[:, 1] - np.min(vertices[:, 1])) / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))

    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_map)
    mesh.textures = [o3d.geometry.Image((texture_image * 255).astype(np.uint8))]

    # Încarcă textura ca imagine Open3D
    texture_image = o3d.geometry.Image((texture_image * 255).astype(np.uint8))

    # Setează textura în material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.albedo_img = texture_image

    mesh_filename = os.path.join(MESHES_FOLDER, f"{base_name}.obj")
    o3d.io.write_triangle_mesh(mesh_filename, mesh, write_vertex_colors=True)
    print(f"Mesh salvat in: {mesh_filename}")

    # Vizualizare mesh cu textură
    o3d.visualization.draw([{"name": "Mesh with Texture", "geometry": mesh, "material": material}])
'''


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
