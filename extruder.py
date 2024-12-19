import cv2
import os

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

import open3d as o3d

PROCESSED_FOLDER = "processed"  # contururile
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


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

    # Aplic alg Canny pt detectare contururi
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Combin contururile detectate cu imaginea color
    edges_colored = cv2.bitwise_and(image, image, mask=edges)

    # Salvează imaginea cu contururile detectate
    output_path = os.path.join(PROCESSED_FOLDER, "edges_" + os.path.basename(image_path))
    cv2.imwrite(output_path, edges_colored)
    return output_path


POSTPROCESSED_FOLDER = "postprocessed"
os.makedirs(POSTPROCESSED_FOLDER, exist_ok=True)


def generate_depth_map(image_path, source_folder="uploads"):
    # Director pentru hărțile de adâncime
    h_map_folder = "depth_maps"
    os.makedirs(h_map_folder, exist_ok=True)

    # Reconstruim calea către imaginea din folderul specificat
    image_path = os.path.join(source_folder, os.path.basename(image_path))

    # Verificăm dacă harta există deja
    depth_map_filename = os.path.basename(image_path).replace(".jpg", "_depth.jpg")
    depth_map_path = os.path.join(h_map_folder, depth_map_filename)
    if os.path.exists(depth_map_path):
        print(f"Harta de adâncime deja există: {depth_map_path}")
        return depth_map_path

    # Transformarile pentru modelul MiDaS
    transform = Compose([
        Resize(384),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])

    # Încarcă modelul MiDaS
    model_type = "DPT_Large"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    # Încarcă imaginea
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


def generate_3d_mesh(depth_map, edge_map=None):
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
            if z > 0:  # Ignorăm punctele fără adâncime
                points.append([x, height - y, z])  # Inversăm coordonata Y
                colors.append([z / depth_scale, z / depth_scale, z / depth_scale])  # Scală de gri pentru culori
    points = np.array(points)
    colors = np.array(colors)

    # Adăugăm punctele marginilor pentru detalii
    if edge_map is not None:
        edge_points = np.argwhere(edge_map > 0)
        for y, x in edge_points:
            points = np.append(points, [[x, y, depth_map[y, x]]], axis=0)
            colors = np.append(colors, [[1, 0, 0]], axis=0)  # Culoare roșie pentru margini
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

    width = x_max - x_min  # Lungimea (W)
    height = y_max - y_min  # Lățimea (H)
    print(f"Dimensiuni model extrdat: width: {width: .2f}, height: {height: .2f}")
    return mesh


def depth_map_to_3d(depth_map_path):
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

    # Generăm mesh-ul 3D
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
    print(f"Dimensiuni imagine originală (textură): {texture_image.shape[1]} x {texture_image.shape[0]} (W x H)")

    # Ensure the texture image has the same dimensions as the depth map
    if (texture_image.shape[1], texture_image.shape[0]) != (depth_map.shape[1], depth_map.shape[0]):
        print("Texture dimensions do not match the depth map dimensions. Resizing texture...")
        texture_image = cv2.resize(texture_image, (depth_map.shape[1], depth_map.shape[0]))
    print(f"Dimensiuni imagine după redimensionare: {texture_image.shape[1]} x {texture_image.shape[0]} (W x H)")
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
    mesh.textures = [texture_o3d]
    mesh.compute_vertex_normals()

    # Visualize the mesh with texture
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    # material.base_color = texture_image
    # material.shader = "defaultLit"
    material.albedo_img = texture_o3d

    o3d.visualization.draw([{"name": "Mesh with Texture", "geometry": mesh, "material": material}])
