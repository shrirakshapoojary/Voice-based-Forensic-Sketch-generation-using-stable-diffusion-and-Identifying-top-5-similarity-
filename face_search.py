import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import faiss
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage.metrics import structural_similarity as ssim


@dataclass
class FaceSearchConfig:
    """Configuration for face search and refinement (Facenet + OpenCV, no dlib)."""

    # Paths for FAISS index and metadata (built once from the iDOC dataset).
    index_path: Path = Path("mugshot_index.faiss")
    meta_path: Path = Path("mugshot_meta.json")

    # Embedding dimensionality for InceptionResnetV1 (Facenet).
    embedding_dim: int = 512


class FaceSearchError(Exception):
    """Custom error for face search pipeline."""


def _load_facenet_models():
    """
    Load MTCNN (face detection + alignment) and InceptionResnetV1 (face embeddings).
    Uses CPU by default; will use CUDA if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet, device


def _refine_and_embed_face(
    image_path: Path, models=None
) -> np.ndarray:
    """
    Load an image, perform face detection/alignment using MTCNN,
    apply illumination correction and noise reduction in OpenCV,
    then compute a 512‑D embedding using InceptionResnetV1.
    """
    if models is None:
        mtcnn, resnet, device = _load_facenet_models()
    else:
        mtcnn, resnet, device = models

    # Read image (BGR) with OpenCV.
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FaceSearchError(f"Unable to read image: {image_path}")

    # Convert to RGB for MTCNN/Facenet.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # MTCNN returns an aligned face tensor (C, H, W) in [0,1].
    face_tensor = mtcnn(rgb)
    if face_tensor is None:
        raise FaceSearchError(f"No face detected in image: {image_path}")

    # Move to CPU numpy for OpenCV-based refinement.
    face_np = face_tensor.permute(1, 2, 0).detach().cpu().numpy()  # HWC, [0,1]
    face_np = (face_np * 255).astype("uint8")

    # Convert to grayscale for normalization and denoising.
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)

    # Illumination correction via histogram equalization.
    gray_eq = cv2.equalizeHist(gray)

    # Noise reduction (edge‑preserving) with bilateral filter.
    denoised = cv2.bilateralFilter(gray_eq, d=9, sigmaColor=75, sigmaSpace=75)

    # Back to RGB tensor in [0,1] for Facenet.
    refined_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    refined_rgb = refined_rgb.astype("float32") / 255.0
    refined_tensor = (
        torch.from_numpy(refined_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # Compute 512‑D embedding.
    with torch.no_grad():
        embedding = resnet(refined_tensor).cpu().numpy()[0].astype("float32")
    return embedding


def _load_faiss_index(cfg: FaceSearchConfig) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not cfg.index_path.exists() or not cfg.meta_path.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata not found. Expected {cfg.index_path} and {cfg.meta_path}."
        )

    index = faiss.read_index(str(cfg.index_path))
    with cfg.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def _compute_ssim_score(query_img: np.ndarray, cand_img: np.ndarray) -> float:
    """
    Compute SSIM between two images after resizing and converting to grayscale.
    """
    # Resize candidate to match query size.
    cand_resized = cv2.resize(cand_img, (query_img.shape[1], query_img.shape[0]))
    q_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(cand_resized, cv2.COLOR_BGR2GRAY)
    score = ssim(q_gray, c_gray)
    return float(score)


def search_similar_faces(
    query_image_path: str, top_k: int = 5, cfg: FaceSearchConfig | None = None
) -> List[Dict[str, Any]]:
    """
    End‑to‑end search:
      1. Refine the generated sketch (alignment, normalization, denoising).
      2. Extract a 128‑D embedding with dlib.
      3. Query a FAISS HNSW index built over iDOC mugshot embeddings.
      4. Re‑rank the retrieved candidates with SSIM and return Top‑K.

    Returns a list of dictionaries:
      [{ "image_path": "...", "distance": float, "ssim": float }, ...]
    """
    if cfg is None:
        cfg = FaceSearchConfig()

    image_path = Path(query_image_path)
    if not image_path.exists():
        raise FaceSearchError(f"Query image does not exist: {image_path}")

    # Load Facenet models once and reuse.
    models = _load_facenet_models()

    # 1–2. Refine and embed query face.
    query_embedding = _refine_and_embed_face(image_path, models=models)
    query_vec = query_embedding.reshape(1, -1).astype("float32")

    # 3. Load FAISS index and metadata, perform ANN search.
    index, meta = _load_faiss_index(cfg)

    # Request more neighbors than needed for a stronger SSIM re‑ranking.
    ann_k = max(top_k * 5, top_k)
    ann_k = min(ann_k, len(meta))
    distances, indices = index.search(query_vec, ann_k)

    # Load the query image (BGR) for SSIM comparisons.
    query_bgr = cv2.imread(str(image_path))
    if query_bgr is None:
        raise FaceSearchError(f"Unable to load query image for SSIM: {image_path}")

    # 4. Compute SSIM scores for retrieved candidates.
    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        cand_info = meta[idx]
        cand_path = Path(cand_info["image_path"])
        cand_img = cv2.imread(str(cand_path))
        if cand_img is None:
            continue
        ssim_score = _compute_ssim_score(query_bgr, cand_img)
        results.append(
            {
                "image_path": str(cand_path),
                "distance": float(dist),
                "ssim": ssim_score,
                "metadata": cand_info,
            }
        )

    # Sort primarily by SSIM (descending), secondarily by distance (ascending).
    results.sort(key=lambda r: (-r["ssim"], r["distance"]))
    return results[:top_k]


def build_mugshot_index(
    dataset_dir: str, cfg: FaceSearchConfig | None = None, max_images: int | None = None
) -> None:
    """
    Build a FAISS HNSW index from all mugshot images in `dataset_dir`.

    This should be run once (offline) from a terminal:
        python -m face_search_build --dataset-dir "path/to/iDOC_dataset"

    The function:
      - walks all images in dataset_dir,
      - refines each face and extracts a 128‑D embedding,
      - stores embeddings in a FAISS HNSW index,
      - saves index + metadata to disk.
    """
    if cfg is None:
        cfg = FaceSearchConfig()

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FaceSearchError(f"Dataset directory does not exist: {dataset_path}")

    # Load Facenet models once.
    models = _load_facenet_models()

    embeddings: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    # Many datasets (including some mugshot releases) use extensionless files.
    # Accept common image extensions *and* files with no extension.
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ""}

    processed = 0
    for img_path in dataset_path.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in supported_ext:
            continue

        if max_images is not None and processed >= max_images:
            break

        try:
            emb = _refine_and_embed_face(img_path, models=models)
            embeddings.append(emb)
            meta.append({"image_path": str(img_path)})
            processed += 1
            # Light progress logging every 500 images.
            if processed % 500 == 0:
                print(f"[build_mugshot_index] Processed {processed} images...")
        except Exception:
            # Skip images where face detection / embedding fails.
            continue

    if not embeddings:
        raise FaceSearchError("No valid embeddings were extracted from the dataset.")

    emb_mat = np.vstack(embeddings).astype("float32")

    # Build FAISS HNSW index.
    index = faiss.IndexHNSWFlat(cfg.embedding_dim, 32)
    index.hnsw.efConstruction = 200
    index.add(emb_mat)

    # Save index and metadata.
    cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.index_path))
    with cfg.meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


