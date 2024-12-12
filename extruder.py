import cv2
import os

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

import open3d as o3d

'''def convert_image_to_grayscale(image_path):
    # Încarcă imaginea în format color
    image = cv2.imread(image_path)

    # Transformă imaginea în alb-negru
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Salvează imaginea alb-negru într-o cale temporară
    output_path = os.path.splitext(image_path)[0] + "_grayscale.jpg"
    cv2.imwrite(output_path, grayscale_image)

    return output_path
'''

'''def adjust_contrast_and_brightness(image_path, contrast=1.5, brightness=50):
    """
    Ajustează contrastul și luminozitatea unei imagini.

    :param image_path: Calea către imaginea de intrare.
    :param contrast: Factorul de contrast (default 1.5).
    :param brightness: Valoarea de luminozitate adăugată (default 50).
    :return: Calea către imaginea ajustată.
    """
    # Încarcă imaginea
    image = cv2.imread(image_path)

    # Ajustează contrastul și luminozitatea
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # Creează folderul "adjusted-contrast" dacă nu există
    adjusted_folder = "adjusted-contrast"
    os.makedirs(adjusted_folder, exist_ok=True)

    # Salvează imaginea ajustată în folderul "adjusted-contrast"
    output_path = os.path.join(adjusted_folder, os.path.basename(image_path).replace(".jpg", "_adjusted.jpg"))
    cv2.imwrite(output_path, adjusted_image)

    return output_path
'''

'''def modif_image(image_path):
    """
    Modifică contrastul și luminozitatea unei imagini la valori maxime.

    :param image_path: Calea către imaginea de intrare.
    :return: Calea către imaginea ajustată.
    """
    # Încarcă imaginea
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Imaginea nu a fost găsită la calea: {image_path}")

    # Setează valori maxime pentru contrast și luminozitate
    contrast = 5.0  # Valoare ridicată pentru contrast
    brightness = 5  # Valoare ridicată pentru luminozitate

    # Ajustează contrastul și luminozitatea
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # Creează folderul "adjusted" dacă nu există
    adjusted_folder = "adjusted"
    os.makedirs(adjusted_folder, exist_ok=True)

    # Salvează imaginea ajustată în folderul "adjusted"
    output_path = os.path.join(
        adjusted_folder, os.path.basename(image_path).replace(".jpg", "_adjusted.jpg")
    )
    cv2.imwrite(output_path, adjusted_image)

    return output_path
'''

PROCESSED_FOLDER = "processed"  # contururile
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def detect_edges(image_path):
    # Verificăm dacă fișierul este deja o imagine procesată (_depth)
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


def overlay_edges_on_original(image_path):
    # Vf daca fis este deja procesat
    output_path = os.path.join(POSTPROCESSED_FOLDER, "overlay_" + os.path.basename(image_path))
    if os.path.exists(output_path):
        print(f"Fisierul suprapus există deja: {output_path}")
        return output_path

    image = cv2.imread(image_path)
    if image is None:
        print(f"Imaginea {image_path} nu a putut fi încărcată.")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Creăm o imagine color cu doar contururile
    edges_colored = cv2.merge([edges, edges, edges])  # Facem imaginea în 3 canale (RGB)

    # Suprapunem contururile peste imaginea originală (folosind alfa pentru transparență, dacă dorim)
    overlay = cv2.addWeighted(image, 0.8, edges_colored, 0.5, 0)

    # Salvăm imaginea suprapusă în folderul /postprocessed
    cv2.imwrite(output_path, overlay)
    print(f"Imaginea suprapusă a fost salvată în: {output_path}")
    return output_path


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
    print(f"Harta de adancime procesata si salvata: {output_path}")

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
    print(f"Număr de puncte: {len(points)}, Culori adăugate: {len(colors)}")

    # Creează un nor de puncte
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Reconstrucție mesh 3D folosind Poisson Surface Reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()

    return mesh


def depth_map_to_3d(depth_map_path):
    """
    Procesează o hartă de adâncime pentru a genera un model 3D.
    """
    # Preprocesăm harta de adâncime
    depth_map = preprocess_depth_map(depth_map_path)

    # Detectăm marginile folosind funcția personalizată detect_edges
    edge_image_path = detect_edges(depth_map_path)
    edge_map = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    print(f"sal {np.unique(edge_map)}")
    # Generăm mesh-ul 3D
    mesh = generate_3d_mesh(depth_map, edge_map)

    # Afișăm mesh-ul
    o3d.visualization.draw_geometries([mesh], window_name="Model 3D Extrudat", width=800, height=600)
