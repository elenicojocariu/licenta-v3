import cv2
import os

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np


'''def extrude_image_to_3d(image_path, extrusion_depth=10):
    # Încarcă imaginea și aplică transformarea 3D
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Creează mesh-ul (aceasta este o funcție exemplu)
    vertices = []
    faces = []
    # (Definește-ți structura aici)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    output_path = "output_model.obj"
    mesh.export(output_path)
    return output_path '''


def convert_image_to_grayscale(image_path):
    # Încarcă imaginea în format color
    image = cv2.imread(image_path)

    # Transformă imaginea în alb-negru
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Salvează imaginea alb-negru într-o cale temporară
    output_path = os.path.splitext(image_path)[0] + "_grayscale.jpg"
    cv2.imwrite(output_path, grayscale_image)

    return output_path


def adjust_contrast_and_brightness(image_path, contrast=1.5, brightness=50):
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

    # Salvează imaginea ajustată
    output_path = os.path.splitext(image_path)[0] + "_adjusted.jpg"
    cv2.imwrite(output_path, adjusted_image)

    return output_path


PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def detect_edges(image_path):
    # Ajustează contrastul și luminozitatea imaginii
    adjusted_image_path = adjust_contrast_and_brightness(image_path)
    # Citește imaginea ajustată
    image = cv2.imread(adjusted_image_path)
    # Convertește imaginea în grayscale pentru detectarea contururilor
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplică algoritmul Canny pentru detectarea contururilor
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
    # Combină contururile detectate cu imaginea color
    edges_colored = cv2.bitwise_and(image, image, mask=edges)
    # Salvează imaginea cu contururile detectate
    output_path = os.path.join(PROCESSED_FOLDER, "edges_" + os.path.basename(image_path))
    cv2.imwrite(output_path, edges_colored)
    return output_path


def generate_depth_map(image_path):

    # Director pentru hărțile de adâncime
    h_map_folder = "depth_maps"
    os.makedirs(h_map_folder, exist_ok=True)

    # Verificăm dacă harta există deja
    depth_map_filename = os.path.basename(image_path).replace(".jpg", "_depth.jpg")
    depth_map_path = os.path.join(h_map_folder, depth_map_filename)
    if os.path.exists(depth_map_path):
        print(f"Harta de adâncime deja există: {depth_map_path}")
        return depth_map_path

    adjusted_image_path = adjust_contrast_and_brightness(image_path)

    # Transformările pentru modelul MiDaS
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
    img = Image.open(adjusted_image_path).convert("RGB")
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


'''def overlay_edges(image_path):
    # Ajustează contrastul și luminozitatea imaginii
    adjusted_image_path = adjust_contrast_and_brightness(image_path)
    # Citește imaginea ajustată
    image = cv2.imread(adjusted_image_path)
    # Detectează marginile
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
    # Suprapune marginile peste imaginea originală
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blended_image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    # Salvează imaginea suprapusă
    output_path = os.path.splitext(image_path)[0] + "_overlay.jpg"
    cv2.imwrite(output_path, blended_image)
    return output_path
'''
