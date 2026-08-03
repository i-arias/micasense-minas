"""
================================================================================
PIPELINE DE PROCESAMIENTO - DETECCIÓN DE MINAS ANTIPERSONA
MicaSense RedEdge-MX - Imágenes Multiespectrales
================================================================================

Autor: Isaac Arias Marín
Descripción: Pipeline completo para calibración radiométrica, alineación de 
             bandas y conversión a reflectancia.

Uso:
    python preprocesamiento.py --ruta_base "RUTA/A/TUS/DATOS" --dia "DIA1"
    
    O modificar CONFIG directamente en el script.
================================================================================
"""

import os
import re
import glob
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
from skimage import filters, morphology
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CONFIGURACIÓN - MODIFICAR SEGÚN TU ESTRUCTURA DE DATOS
# ==============================================================================

CONFIG = {
    # Ruta base del proyecto (modificar según tu estructura)
    'RUTA_BASE': r"C:\Users\isaac\OneDrive\Documents\Camara\micasense_tool",
    
    # Carpeta que contiene las imágenes del panel de calibración
    'CARPETA_PANEL': "PANEL_INICIO",
    
    # Lista de carpetas objetivo a procesar
    'CARPETAS_OBJETIVO': [
        "MINA_1CM_1",
        "MINA_1CM_2",
        "MINA_3CM_1",
        "MINA_3CM_2",
        "MINA_5CM_1",
        "MINA_5CM_2",
        "MINA_7CM_1",
        "MINA_7CM_2",
        "ZONA_CONTROL",
        "ZONA_CONTROL_2",
        "ZONA_CONTROL_3",
        "ZONA_CONTROL_4",
    ],
    
    # Valores de albedo del panel CRP para cada banda
    # [Blue, Green, Red, RedEdge, NIR]
    'PANEL_ALBEDO': [0.490, 0.491, 0.491, 0.488, 0.490],
    
    # Recorte para eliminar bordes negros generados por la alineación de bandas
    # Los artefactos son asimétricos: izquierda y abajo son los más afectados
    'RECORTE': {
        'top': 20,
        'bottom': 60,
        'left': 35,
        'right': 20
    },
    
    # Nombres de las bandas
    'BANDAS_NOMBRES': ['Blue (475nm)', 'Green (560nm)', 'Red (668nm)', 
                       'Red Edge (717nm)', 'NIR (842nm)']
}


# ==============================================================================
# FUNCIONES DE CARGA DE IMÁGENES
# ==============================================================================

def cargar_imagenes_banda(ruta_carpeta):
    """
    Carga las 5 bandas de una captura MicaSense desde una carpeta.
    
    Args:
        ruta_carpeta: Ruta a la carpeta con las 5 imágenes .tif
        
    Returns:
        Lista de rutas a los archivos ordenados por banda (1-5)
    """
    patron = os.path.join(ruta_carpeta, "*_?.tif")
    archivos = sorted(glob.glob(patron))
    
    if len(archivos) != 5:
        # Intentar patrón alternativo
        patron = os.path.join(ruta_carpeta, "*.tif")
        archivos = sorted(glob.glob(patron))
        
    if len(archivos) != 5:
        raise ValueError(f"Se esperaban 5 bandas, se encontraron {len(archivos)} en {ruta_carpeta}")
    
    # Ordenar por el último número en el nombre (soporta band1, _1, etc.)
    archivos_ordenados = sorted(archivos, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]))
    
    return archivos_ordenados


# ==============================================================================
# FUNCIONES DE CALIBRACIÓN
# ==============================================================================

def detectar_panel_robusto(radiance, banda_num):
    """
    Detecta el panel CRP en una imagen usando múltiples métodos robustos.
    """
    try:
        # Normalizar imagen para procesamiento
        img_norm = (radiance - radiance.min()) / (radiance.max() - radiance.min() + 1e-10)
        
        # MÉTODO 1: Segmentación por umbral Otsu
        try:
            threshold = filters.threshold_otsu(img_norm)
            mask = img_norm > threshold
            
            # Limpiar máscara
            mask = morphology.remove_small_objects(mask, min_size=1000)
            mask = morphology.remove_small_holes(mask, area_threshold=1000)
            
            # Encontrar la región conectada más grande
            labeled, num_features = ndimage.label(mask)
            
            if num_features > 0:
                sizes = ndimage.sum(mask, labeled, range(num_features + 1))
                largest_label = np.argmax(sizes[1:]) + 1
                panel_mask = labeled == largest_label

                total_pixeles = radiance.size
                n_pixeles = np.sum(panel_mask)
                # El panel ocupa máximo ~15% de la imagen a 1m de altura
                if 500 < n_pixeles < total_pixeles * 0.15:
                    panel_radiance = np.mean(radiance[panel_mask])
                    print(f"      → Método Otsu: {n_pixeles} píxeles detectados")
                    return panel_radiance
                elif n_pixeles >= total_pixeles * 0.15:
                    print(f"      ⚠️  Otsu detectó región demasiado grande ({n_pixeles} px, {n_pixeles/total_pixeles*100:.1f}%) — descartando")
        except Exception as e:
            print(f"      ⚠️  Método Otsu falló: {str(e)}")
        
        # MÉTODO 2: Top percentil
        threshold_low = np.percentile(radiance, 90)
        mask_percentile = radiance >= threshold_low
        
        if np.sum(mask_percentile) > 100:
            panel_radiance = np.mean(radiance[mask_percentile])
            print(f"      → Método percentil: {np.sum(mask_percentile)} píxeles")
            return panel_radiance
        
        # MÉTODO 3: Región central
        h, w = radiance.shape
        cy, cx = h // 2, w // 2
        size = min(h, w) // 4
        
        panel_region = radiance[cy-size:cy+size, cx-size:cx+size]
        panel_radiance = np.mean(panel_region)
        print(f"      → Método región central: área {size*2}x{size*2}")
        
        return panel_radiance
        
    except Exception as e:
        print(f"      ⚠️  Todos los métodos fallaron: {str(e)}")
        return np.mean(radiance)


def detectar_mascara_panel(ruta_panel_archivos):
    """
    Detecta la máscara del panel usando la banda Red (índice 2), que es la más
    confiable con Otsu. Devuelve la máscara para aplicar a todas las bandas.
    """
    REF_BANDA = 2  # Red
    with rasterio.open(ruta_panel_archivos[REF_BANDA]) as src:
        img_ref = src.read(1).astype(np.float32)

    img_norm = (img_ref - img_ref.min()) / (img_ref.max() - img_ref.min() + 1e-10)
    threshold = filters.threshold_otsu(img_norm)
    mask = img_norm > threshold
    mask = morphology.remove_small_objects(mask, min_size=1000)
    mask = morphology.remove_small_holes(mask, area_threshold=1000)

    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None

    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    panel_mask = labeled == largest_label

    n = np.sum(panel_mask)
    total = img_ref.size
    if n < 500 or n > total * 0.15:
        return None

    print(f"    → Máscara del panel detectada en banda Red: {n} píxeles ({n/total*100:.1f}%)")
    return panel_mask


def calcular_irradiancia_panel(ruta_panel, panel_albedo):
    """
    Calcula la irradiancia solar para cada banda usando el panel de calibración.
    Usa la posición del panel detectada en la banda Red para todas las bandas.
    """
    print(f"\n[CALIBRACIÓN] Procesando panel de calibración...")

    archivos_panel = cargar_imagenes_banda(ruta_panel)

    # Detectar máscara del panel una sola vez usando la banda Red
    panel_mask = detectar_mascara_panel(archivos_panel)

    irradiancias = []

    for i, archivo in enumerate(archivos_panel):
        with rasterio.open(archivo) as src:
            img_raw = src.read(1).astype(np.float32)

        print(f"    → Banda {i+1}: Calculando irradiancia...")

        if panel_mask is not None:
            panel_radiance = np.mean(img_raw[panel_mask])
            print(f"      → Máscara compartida: {np.sum(panel_mask)} píxeles")
        else:
            panel_radiance = detectar_panel_robusto(img_raw, i+1)

        # Calcular irradiancia: E = L * π / ρ
        irradiance = panel_radiance * np.pi / panel_albedo[i]
        irradiancias.append(irradiance)
        print(f"  ✓ Banda {i+1}: Irradiancia = {irradiance:.2f}")
    
    return irradiancias


# ==============================================================================
# FUNCIONES DE ALINEACIÓN
# ==============================================================================

def validar_homografia(H, umbral_escala=2.0, umbral_shear=0.5):
    """Valida que una homografía sea razonable y no degenerada."""
    try:
        H = H / H[2, 2]
        
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if det <= 0 or det > umbral_escala**2 or det < 1/umbral_escala**2:
            return False
        
        if abs(H[2, 0]) > 0.001 or abs(H[2, 1]) > 0.001:
            return False
        
        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
        
        if scale_x > umbral_escala or scale_x < 1/umbral_escala:
            return False
        if scale_y > umbral_escala or scale_y < 1/umbral_escala:
            return False
        
        return True
    except Exception:
        return False


def alinear_con_sift(img_to_align, ref_img, img_enhanced, ref_enhanced, banda_num):
    """
    Alineación usando SIFT + RANSAC optimizado para MicaSense.
    """
    try:
        sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        kp1, des1 = sift.detectAndCompute(ref_enhanced, None)
        kp2, des2 = sift.detectAndCompute(img_enhanced, None)
        
        if des1 is None or des2 is None:
            return None
        
        # Matching con FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Ratio test de Lowe
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        result = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0, maxIters=5000)
        
        if result is None or result[0] is None:
            return None
        
        H, mask = result[0], result[1]
        
        if not validar_homografia(H):
            return None
        
        inliers = np.sum(mask)
        print(f"    ✓ Banda {banda_num}: {inliers} inliers ({inliers/len(good_matches)*100:.1f}%)")
        
        img_aligned = cv2.warpPerspective(
            img_to_align, H,
            (img_to_align.shape[1], img_to_align.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return img_aligned
        
    except Exception as e:
        print(f"    ❌ Banda {banda_num}: Error SIFT: {str(e)}")
        return None


def alinear_con_ecc(img_to_align, img_norm, ref_norm, banda_num):
    """Método de alineación ECC como fallback."""
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
        warp_init = np.eye(2, 3, dtype=np.float32)
        
        _, warp_matrix = cv2.findTransformECC(
            ref_norm, img_norm, warp_init,
            cv2.MOTION_EUCLIDEAN, criteria
        )
        
        img_aligned = cv2.warpAffine(
            img_to_align, warp_matrix,
            (img_to_align.shape[1], img_to_align.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        print(f"    ✓ Banda {banda_num}: Alineación ECC exitosa")
        return img_aligned
        
    except Exception as e:
        return None


# ==============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ==============================================================================

def procesar_captura(ruta_objetivo, irradiancias, nombre_salida, ruta_salida, recorte=None):
    """
    Procesa una captura: alineación de bandas y conversión a reflectancia.
    """
    archivos_objetivo = cargar_imagenes_banda(ruta_objetivo)
    
    print(f"  → Cargando bandas...")
    imagenes_raw = []
    metas = []
    
    for archivo in archivos_objetivo:
        with rasterio.open(archivo) as src:
            img = src.read(1).astype(np.float32)
            imagenes_raw.append(img)
            metas.append(src.meta.copy())
    
    # Banda Red (índice 2) como referencia
    ref_idx = 2
    ref_img = imagenes_raw[ref_idx]
    
    print(f"  → Alineando bandas (referencia: banda {ref_idx + 1} - Red)...")
    
    imagenes_aligned = []
    
    for i, img in enumerate(imagenes_raw):
        if i == ref_idx:
            imagenes_aligned.append(img)
            continue
        
        # Normalizar para alineación
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ref_norm = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_norm)
        ref_enhanced = clahe.apply(ref_norm)
        
        # Intentar SIFT primero
        img_aligned = alinear_con_sift(img, ref_img, img_enhanced, ref_enhanced, i+1)
        
        # Fallback a ECC
        if img_aligned is None:
            img_aligned = alinear_con_ecc(img, img_norm, ref_norm, i+1)
        
        # Si todo falla, usar original
        if img_aligned is None:
            print(f"    ⚠️  Banda {i+1}: Usando imagen sin alinear")
            img_aligned = img
        
        imagenes_aligned.append(img_aligned)
    
    # Aplicar recorte si está configurado
    if recorte:
        t, b = recorte['top'], recorte['bottom']
        l, r = recorte['left'], recorte['right']
        h, w = imagenes_aligned[0].shape
        imagenes_aligned = [img[t:h-b, l:w-r] for img in imagenes_aligned]
    
    # Convertir a reflectancia
    print(f"  → Convirtiendo a reflectancia...")
    h, w = imagenes_aligned[0].shape
    im_reflectance = np.zeros((h, w, 5), dtype=np.float32)
    
    for i, img_aligned in enumerate(imagenes_aligned):
        reflectance = img_aligned * np.pi / irradiancias[i]
        reflectance = np.clip(reflectance, 0, 1)
        im_reflectance[:, :, i] = reflectance
    
    # Guardar como GeoTIFF multibanda
    os.makedirs(ruta_salida, exist_ok=True)
    archivo_salida = os.path.join(ruta_salida, f"{nombre_salida}_calibrado.tif")
    
    meta = metas[ref_idx].copy()
    meta.update({
        'height': h,
        'width': w,
        'count': 5,
        'dtype': 'float32',
        'compress': 'lzw'
    })
    
    print(f"  → Guardando: {archivo_salida}")
    with rasterio.open(archivo_salida, 'w', **meta) as dst:
        for i in range(5):
            dst.write(im_reflectance[:, :, i], i + 1)
            dst.set_band_description(i + 1, CONFIG['BANDAS_NOMBRES'][i])
    
    return archivo_salida


# ==============================================================================
# MAIN
# ==============================================================================

def main(ruta_datos, carpeta_salida="PROCESADAS"):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    """
    print("=" * 80)
    print("PIPELINE DE PREPROCESAMIENTO - IMÁGENES MULTIESPECTRALES")
    print("=" * 80)
    
    ruta_salida = os.path.join(ruta_datos, carpeta_salida)
    
    # 1. Calcular irradiancias del panel
    ruta_panel = os.path.join(ruta_datos, CONFIG['CARPETA_PANEL'])
    
    if not os.path.exists(ruta_panel):
        print(f"❌ Error: No se encontró carpeta de panel: {ruta_panel}")
        return []
    
    irradiancias = calcular_irradiancia_panel(ruta_panel, CONFIG['PANEL_ALBEDO'])
    
    # 2. Procesar cada carpeta objetivo
    print(f"\n{'='*80}")
    print("PROCESANDO CAPTURAS")
    print(f"{'='*80}\n")
    
    imagenes_procesadas = []
    
    for carpeta in tqdm(CONFIG['CARPETAS_OBJETIVO'], desc="Procesando"):
        ruta_objetivo = os.path.join(ruta_datos, carpeta)
        
        if not os.path.exists(ruta_objetivo):
            print(f"\n⚠️  Omitiendo {carpeta} (no existe)")
            continue
        
        print(f"\n[PROCESANDO] {carpeta}")
        
        try:
            archivo = procesar_captura(
                ruta_objetivo=ruta_objetivo,
                irradiancias=irradiancias,
                nombre_salida=carpeta,
                ruta_salida=ruta_salida,
                recorte=CONFIG['RECORTE']
            )
            imagenes_procesadas.append((carpeta, archivo))
            print(f"  ✅ Completado")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETADO: {len(imagenes_procesadas)}/{len(CONFIG['CARPETAS_OBJETIVO'])} imágenes")
    print(f"{'='*80}\n")
    
    return imagenes_procesadas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocesamiento de imágenes MicaSense')
    parser.add_argument('--ruta', type=str, help='Ruta a la carpeta con los datos del día')
    parser.add_argument('--salida', type=str, default='PROCESADAS', help='Nombre carpeta de salida')
    
    args = parser.parse_args()
    
    if args.ruta:
        main(args.ruta, args.salida)
    else:
        # Usar ruta de ejemplo
        print("Uso: python preprocesamiento.py --ruta 'RUTA/A/DIA1/SET_01'")
        print("\nO modificar CONFIG en el script y ejecutar directamente.")
