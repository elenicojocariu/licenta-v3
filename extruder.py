import cv2
import os

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

import open3d as o3d

PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MESHES_FOLDER = "meshes"
os.makedirs(MESHES_FOLDER, exist_ok=True)


def detect_edges(image_path):
    if "_depth" in image_path:
        print(f"Imaginea {image_path} a fost deja procesată, o ignorăm.")
        return None

    output_path = os.path.join(PROCESSED_FOLDER, "edges_" + os.path.basename(image_path))
    if os.path.exists(output_path):
        print(f"Fisierul pentru margini exista deja: {output_path}")
        return output_path

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Combin contururile detectate cu imaginea color
    edges_colored = cv2.bitwise_and(image, image, mask=edges)

    output_path = os.path.join(PROCESSED_FOLDER, "edges_" + os.path.basename(image_path))
    cv2.imwrite(output_path, edges_colored)
    return output_path


POSTPROCESSED_FOLDER = "postprocessed"
os.makedirs(POSTPROCESSED_FOLDER, exist_ok=True)


def generate_depth_map(image_path, source_folder="uploads"):
    h_map_folder = "depth_maps"
    os.makedirs(h_map_folder, exist_ok=True)

    image_path = os.path.join(source_folder, os.path.basename(image_path))

    depth_map_filename = os.path.basename(image_path).replace(".jpg", "_depth.jpg")
    depth_map_path = os.path.join(h_map_folder, depth_map_filename)
    if os.path.exists(depth_map_path):
        print(f"Harta de adâncime deja există: {depth_map_path}")
        return depth_map_path

    transform = Compose([
        Resize(384),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    model_type = "DPT_Large"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    input_batch = transform(img).unsqueeze(0)

    # Generăm harta de adâncime
    with torch.no_grad():
        prediction = model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()

    # Normalizarea și salvarea hărții
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(depth_map_path, depth_map_normalized)
    print(f"Harta de adâncime generată și salvată: {depth_map_path}")

    return depth_map_path


def preprocess_depth_map(depth_map_path):
    """
    Curăță și îmbunătățește harta de adâncime prin aplicarea unui filtru Gaussian.
    """
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Nu s-a putut încărca harta de adâncime: {depth_map_path}")

    # Aplicăm un filtru Gaussian pentru a reduce zgomotul
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    output_path = depth_map_path.replace("depth_maps", "processed_dept_maps")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, smoothed_depth_map)

    # Normalizăm valorile între 0 și 1
    normalized_depth_map = smoothed_depth_map / 255.0

    return normalized_depth_map


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
def create_3d_mesh_with_texture(image_path, depth_map_path, z_scale=1.0):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertim la RGB

    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Nu s-a putut incarca harta de adancime: {depth_map_path}")

    # Normalizez adancimea intre 0 si o valoare max
    depth_map = depth_map.astype(np.float32) / 255.0
    depth_map *= z_scale*50.0
    h, w = depth_map.shape

    # Redimensionăm imaginea originală să corespundă dimensiunilor modelului 3D
    original_image_resized = cv2.resize(original_image, (w, h))
    print(f"Imagine originală redimensionată la: {w} x {h}")

    # Creează puncte (vertices) din imagine și harta de adâncime
    vertices = []
    colors = []
    for y in range(h):
        for x in range(w):
            z = depth_map[y, x]
            vertices.append((x, h - y - 1, z))
            colors.append(original_image_resized[y, x] / 255.0)

    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    triangles = []
    for y in range(h - 1):
        for x in range(w - 1):
            v0 = y * w + x
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            triangles.append((v0, v2, v1))
            triangles.append((v1, v2, v3))

    triangles = np.array(triangles, dtype=np.int32)

    # Creează mesh-ul 3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    height, width, _ = original_image.shape

    #print(f"Dimensiuni imagine originala: {width} x {height} (width x height)")
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    print(f"Dimensiuni model 3D extrudat: width: {x_max - x_min:.2f}, height: {y_max - y_min:.2f}")

    # Verificăm fața triunghiurilor și o corectăm dacă e necesar
    mesh.compute_triangle_normals()
    mesh.orient_triangles()

    # Salvează mesh-ul
    mesh_path = os.path.join(MESHES_FOLDER, "textured_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh-ul 3D cu textură a fost salvat: {mesh_path}")

    # Vizualizează mesh-ul
    o3d.visualization.draw_geometries([mesh])
