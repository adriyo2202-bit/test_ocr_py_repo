import os
import sys
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
 
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
 
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    warnings.warn("EasyOCR not available. Install: pip install easyocr")
 
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    warnings.warn("pytesseract not available. Install: pip install pytesseract")
 
# ─── Logging ──────────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ocr_pipeline")
 
 
# ─── Data Structures ──────────────────────────────────────────────────────────
 
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
 
    @property
    def width(self) -> int:
        return self.x2 - self.x1
 
    @property
    def height(self) -> int:
        return self.y2 - self.y1
 
    @property
    def area(self) -> int:
        return self.width * self.height
 
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
 
 
@dataclass
class TextBlock:
    text: str
    confidence: float           # 0.0 – 1.0
    bbox: BoundingBox
    engine: str                 # "easyocr" | "tesseract" | "ensemble"
    line_number: int = 0
    block_type: str = "text"    # "text" | "table" | "header" | "footer"
    language: str = "en"
 
 
@dataclass
class OCRResult:
    raw_text: str
    structured_blocks: List[TextBlock]
    overall_confidence: float
    processing_time_ms: float
    image_path: str
    image_size: Tuple[int, int]
    preprocessing_applied: List[str]
    engine_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
 
    def to_json(self, indent: int = 2) -> str:
        d = asdict(self)
        # BoundingBox is a nested dataclass — already handled by asdict
        return json.dumps(d, ensure_ascii=False, indent=indent)
 
    def to_markdown(self) -> str:
        lines = [
            f"# OCR Result",
            f"**Image:** `{self.image_path}`",
            f"**Confidence:** {self.overall_confidence:.1%}",
            f"**Engine:** {self.engine_used}",
            f"**Time:** {self.processing_time_ms:.0f} ms",
            f"**Preprocessing:** {', '.join(self.preprocessing_applied) or 'none'}",
            "",
            "---",
            "",
            "## Extracted Text",
            "",
            self.raw_text,
        ]
        return "\n".join(lines)
 
 
# ─── Image Preprocessor ───────────────────────────────────────────────────────
 
class ImagePreprocessor:
    """
    Applies a configurable chain of CV preprocessing steps to maximise OCR
    accuracy.  Each step is logged and its name recorded in the result.
    """
 
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.deskew            = cfg.get("deskew", True)
        self.denoise           = cfg.get("denoise", True)
        self.contrast_enhance  = cfg.get("contrast_enhance", True)
        self.binarize          = cfg.get("binarize", False)   # off by default — EasyOCR prefers colour
        self.upscale_threshold = cfg.get("upscale_threshold", 1000)  # px on shortest side
        self.upscale_factor    = cfg.get("upscale_factor", 2.0)
        self.border_remove     = cfg.get("border_remove", True)
        self.shadow_remove     = cfg.get("shadow_remove", True)
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def process(self, img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Run all enabled preprocessing steps. Returns (processed_img, step_names)."""
        steps: List[str] = []
 
        img = self._fix_orientation(img, steps)
        img = self._upscale_if_small(img, steps)
        if self.shadow_remove:
            img = self._remove_shadows(img, steps)
        if self.denoise:
            img = self._denoise(img, steps)
        if self.contrast_enhance:
            img = self._enhance_contrast(img, steps)
        if self.deskew:
            img = self._deskew(img, steps)
        if self.border_remove:
            img = self._remove_borders(img, steps)
        if self.binarize:
            img = self._binarize(img, steps)
 
        return img, steps
 
    # ── Individual Steps ──────────────────────────────────────────────────────
 
    @staticmethod
    def _fix_orientation(img: np.ndarray, steps: List[str]) -> np.ndarray:
        """Correct EXIF orientation."""
        try:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            exif = pil._getexif()
            if exif:
                for tag, val in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        rotation = {3: 180, 6: 270, 8: 90}.get(val)
                        if rotation:
                            pil = pil.rotate(rotation, expand=True)
                            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                            steps.append(f"exif_rotation_{rotation}deg")
        except Exception:
            pass
        return img
 
    def _upscale_if_small(self, img: np.ndarray, steps: List[str]) -> np.ndarray:
        h, w = img.shape[:2]
        if min(h, w) < self.upscale_threshold:
            scale = self.upscale_factor
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            steps.append(f"upscale_{scale}x")
        return img
 
    @staticmethod
    def _remove_shadows(img: np.ndarray, steps: List[str]) -> np.ndarray:
        """Normalize uneven lighting / shadows via morphological top-hat."""
        rgb_planes = cv2.split(img)
        result_planes = []
        for plane in rgb_planes:
            dilated  = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img   = cv2.medianBlur(dilated, 21)
            diff     = 255 - cv2.absdiff(plane, bg_img)
            norm     = cv2.normalize(diff, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX)
            result_planes.append(norm)
        result = cv2.merge(result_planes)
        steps.append("shadow_removal")
        return result
 
    @staticmethod
    def _denoise(img: np.ndarray, steps: List[str]) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        steps.append("nlm_denoise")
        return denoised
 
    @staticmethod
    def _enhance_contrast(img: np.ndarray, steps: List[str]) -> np.ndarray:
        """CLAHE on L-channel in LAB colour space."""
        lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        lab   = cv2.merge([l, a, b])
        img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        steps.append("clahe_contrast")
        return img
 
    @staticmethod
    def _deskew(img: np.ndarray, steps: List[str]) -> np.ndarray:
        """Detect and correct skew using Hough line transform."""
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines  = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                  minLineLength=100, maxLineGap=10)
        if lines is None:
            return img
 
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)
 
        if not angles:
            return img
 
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img          # negligible skew
 
        h, w = img.shape[:2]
        M   = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        steps.append(f"deskew_{median_angle:.1f}deg")
        return img
 
    @staticmethod
    def _remove_borders(img: np.ndarray, steps: List[str]) -> np.ndarray:
        """Crop away solid-colour borders / scan artifacts."""
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords  = cv2.findNonZero(mask)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y + h, x:x + w]
        # Only accept if the crop is not too aggressive
        orig_area = img.shape[0] * img.shape[1]
        if cropped.shape[0] * cropped.shape[1] > 0.5 * orig_area:
            steps.append("border_remove")
            return cropped
        return img
 
    @staticmethod
    def _binarize(img: np.ndarray, steps: List[str]) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        steps.append("adaptive_binarize")
        return img
 
 
# ─── OCR Engines ──────────────────────────────────────────────────────────────
 
class EasyOCREngine:
    """Wrapper around EasyOCR with caching of the reader object."""
 
    _reader_cache: Dict[str, "easyocr.Reader"] = {}
 
    def __init__(self, languages: List[str], gpu: bool = False):
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR is not installed.")
        self.languages = languages
        self.gpu = gpu
        self._reader = self._get_reader(languages, gpu)
 
    @classmethod
    def _get_reader(cls, languages: List[str], gpu: bool) -> "easyocr.Reader":
        key = "_".join(sorted(languages)) + f"_gpu{gpu}"
        if key not in cls._reader_cache:
            logger.info("Initialising EasyOCR reader (first-time model download may take a moment)…")
            cls._reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
        return cls._reader_cache[key]
 
    def read(self, img: np.ndarray) -> List[TextBlock]:
        results = self._reader.readtext(
            img,
            detail=1,
            paragraph=False,
            batch_size=4,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5,
        )
 
        blocks: List[TextBlock] = []
        for i, (bbox_pts, text, confidence) in enumerate(results):
            if not text.strip():
                continue
            pts = np.array(bbox_pts, dtype=int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=float(confidence),
                bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                engine="easyocr",
                line_number=i,
            ))
        return blocks
 
 
class TesseractEngine:
    """Wrapper around Tesseract with PSM/OEM selection."""
 
    PSM_AUTO       = 3   # Fully automatic page segmentation
    PSM_SINGLE_BLOCK = 6 # Single uniform block of text
    PSM_SINGLE_LINE  = 7 # Single text line
 
    def __init__(self, lang: str = "eng", psm: int = 3):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract is not installed.")
        self.lang = lang
        self.psm  = psm
 
    def read(self, img: np.ndarray) -> List[TextBlock]:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        config  = f"--oem 3 --psm {self.psm}"
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return []
 
        blocks: List[TextBlock] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 0:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append(TextBlock(
                text=text,
                confidence=conf / 100.0,
                bbox=BoundingBox(x, y, x + w, y + h),
                engine="tesseract",
                line_number=data["line_num"][i],
                block_type="text",
            ))
        return blocks
 
 
# ─── Layout Reconstructor ─────────────────────────────────────────────────────
 
class LayoutReconstructor:
    """
    Re-assembles raw TextBlocks (unordered bounding-box detections) into
    natural reading order: top-to-bottom, left-to-right, with line grouping.
    """
 
    def __init__(self, line_y_tolerance: float = 0.5):
        """
        line_y_tolerance: fraction of median block height used to decide
                          whether two blocks sit on the same line.
        """
        self.line_y_tolerance = line_y_tolerance
 
    def reconstruct(self, blocks: List[TextBlock]) -> str:
        if not blocks:
            return ""
 
        # Sort top-to-bottom by vertical centre
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.center[1])
 
        # Estimate typical text height
        heights = [b.bbox.height for b in sorted_blocks if b.bbox.height > 0]
        median_h = float(np.median(heights)) if heights else 20.0
        tol = median_h * self.line_y_tolerance
 
        # Group into rows
        rows: List[List[TextBlock]] = []
        current_row: List[TextBlock] = [sorted_blocks[0]]
        ref_y = sorted_blocks[0].bbox.center[1]
 
        for blk in sorted_blocks[1:]:
            cy = blk.bbox.center[1]
            if abs(cy - ref_y) <= tol:
                current_row.append(blk)
            else:
                rows.append(current_row)
                current_row = [blk]
                ref_y = cy
        rows.append(current_row)
 
        # Within each row sort left-to-right
        lines = []
        for row in rows:
            row.sort(key=lambda b: b.bbox.x1)
            lines.append(" ".join(b.text for b in row))
 
        return "\n".join(lines)
 
 
# ─── Post-Processor ───────────────────────────────────────────────────────────
 
class TextPostProcessor:
    """Cleans and normalises raw OCR output."""
 
    # Common OCR confusion pairs (add domain-specific corrections here)
    SUBSTITUTIONS: Dict[str, str] = {
        "0": "O",   # only applied in obvious alpha contexts — see below
        "|": "I",
        "l": "1",   # numerics context
    }
 
    def clean(self, text: str) -> str:
        if not text:
            return text
        # Collapse excessive whitespace
        lines = []
        for line in text.splitlines():
            cleaned = " ".join(line.split())
            if cleaned:
                lines.append(cleaned)
        text = "\n".join(lines)
        # Remove lone noise characters on their own line
        filtered = []
        for line in text.splitlines():
            if len(line.strip()) >= 1:
                filtered.append(line)
        return "\n".join(filtered)
 
    def fix_encoding(self, text: str) -> str:
        """Attempt to fix mojibake / encoding issues."""
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text
 
 
# ─── Confidence Scorer ────────────────────────────────────────────────────────
 
class ConfidenceScorer:
 
    @staticmethod
    def score(blocks: List[TextBlock]) -> float:
        if not blocks:
            return 0.0
        weighted = sum(b.confidence * len(b.text) for b in blocks)
        total_chars = sum(len(b.text) for b in blocks)
        return weighted / total_chars if total_chars else 0.0
 
    @staticmethod
    def quality_label(score: float) -> str:
        if score >= 0.90:
            return "excellent"
        if score >= 0.75:
            return "good"
        if score >= 0.55:
            return "fair"
        return "poor"
 
 
# ─── Main Pipeline ────────────────────────────────────────────────────────────
 
class OCRPipeline:
    """
    Orchestrates the full OCR workflow:
      1. Image loading & validation
      2. Preprocessing
      3. Primary engine (EasyOCR) → fallback (Tesseract) if confidence low
      4. Optional ensemble (merge both engines)
      5. Layout reconstruction
      6. Post-processing
      7. Structured result packaging
    """
 
    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False,
        preprocess_config: Optional[Dict] = None,
        confidence_threshold: float = 0.60,
        ensemble: bool = False,
        tesseract_lang: str = "eng",
        save_debug_images: bool = False,
        output_dir: str = "ocr_output",
    ):
        self.languages            = languages or ["en"]
        self.gpu                  = gpu
        self.confidence_threshold = confidence_threshold
        self.ensemble             = ensemble
        self.tesseract_lang       = tesseract_lang
        self.save_debug_images    = save_debug_images
        self.output_dir           = Path(output_dir)
 
        self.preprocessor  = ImagePreprocessor(preprocess_config)
        self.layout        = LayoutReconstructor()
        self.postproc      = TextPostProcessor()
        self.scorer        = ConfidenceScorer()
 
        # Lazy-init engines
        self._easyocr_engine: Optional[EasyOCREngine] = None
        self._tesseract_engine: Optional[TesseractEngine] = None
 
        if self.save_debug_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Engine accessors ──────────────────────────────────────────────────────
 
    @property
    def easyocr(self) -> EasyOCREngine:
        if self._easyocr_engine is None:
            self._easyocr_engine = EasyOCREngine(self.languages, self.gpu)
        return self._easyocr_engine
 
    @property
    def tesseract(self) -> TesseractEngine:
        if self._tesseract_engine is None:
            self._tesseract_engine = TesseractEngine(self.tesseract_lang)
        return self._tesseract_engine
 
    # ── Image Loading ─────────────────────────────────────────────────────────
 
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
 
        ext = Path(path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                # Fallback via PIL (handles exotic formats)
                pil = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
 
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
 
    # ── Debug helpers ─────────────────────────────────────────────────────────
 
    def _save_debug(self, img: np.ndarray, blocks: List[TextBlock], stem: str):
        vis = img.copy()
        for blk in blocks:
            b = blk.bbox
            color = (0, 255, 0) if blk.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            label = f"{blk.confidence:.2f}"
            cv2.putText(vis, label, (b.x1, max(b.y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out_path = self.output_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(out_path), vis)
        logger.info(f"Debug image saved: {out_path}")
 
    # ── Core pipeline ─────────────────────────────────────────────────────────
 
    def run(self, image_path: str) -> OCRResult:
        t0 = time.perf_counter()
        logger.info(f"Processing: {image_path}")
 
        # 1. Load
        img_raw = self.load_image(image_path)
        h, w    = img_raw.shape[:2]
        logger.info(f"Image size: {w}×{h} px")
 
        # 2. Preprocess
        img_proc, prep_steps = self.preprocessor.process(img_raw)
        logger.info(f"Preprocessing: {prep_steps}")
 
        if self.save_debug_images:
            stem = Path(image_path).stem
            cv2.imwrite(str(self.output_dir / f"{stem}_preprocessed.jpg"), img_proc)
 
        # 3. Primary: EasyOCR
        blocks: List[TextBlock] = []
        engine_used = "none"
 
        if EASYOCR_AVAILABLE:
            try:
                blocks = self.easyocr.read(img_proc)
                engine_used = "easyocr"
                conf = self.scorer.score(blocks)
                logger.info(f"EasyOCR: {len(blocks)} blocks, confidence={conf:.1%}")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
 
        # 4. Fallback / Ensemble: Tesseract
        if TESSERACT_AVAILABLE:
            primary_conf = self.scorer.score(blocks)
            need_fallback = primary_conf < self.confidence_threshold or not blocks
            if need_fallback or self.ensemble:
                try:
                    tess_blocks = self.tesseract.read(img_proc)
                    tess_conf   = self.scorer.score(tess_blocks)
                    logger.info(f"Tesseract: {len(tess_blocks)} blocks, confidence={tess_conf:.1%}")
 
                    if self.ensemble and blocks:
                        # Merge: prefer EasyOCR but fill gaps with Tesseract
                        blocks = self._ensemble_merge(blocks, tess_blocks)
                        engine_used = "ensemble"
                    elif tess_conf > primary_conf:
                        blocks = tess_blocks
                        engine_used = "tesseract"
                except Exception as e:
                    logger.warning(f"Tesseract fallback failed: {e}")
 
        if not blocks:
            logger.warning("No text detected by any engine.")
 
        # 5. Save debug visualisation
        if self.save_debug_images and blocks:
            self._save_debug(img_proc, blocks, Path(image_path).stem)
 
        # 6. Layout reconstruction
        raw_text = self.layout.reconstruct(blocks)
 
        # 7. Post-processing
        clean_text = self.postproc.clean(raw_text)
 
        # 8. Final scoring
        overall_conf = self.scorer.score(blocks)
        quality      = self.scorer.quality_label(overall_conf)
 
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Done in {elapsed_ms:.0f} ms | "
            f"engine={engine_used} | confidence={overall_conf:.1%} ({quality})"
        )
 
        return OCRResult(
            raw_text=clean_text,
            structured_blocks=blocks,
            overall_confidence=overall_conf,
            processing_time_ms=elapsed_ms,
            image_path=image_path,
            image_size=(w, h),
            preprocessing_applied=prep_steps,
            engine_used=engine_used,
            metadata={
                "quality": quality,
                "block_count": len(blocks),
                "languages": self.languages,
                "gpu": self.gpu,
            },
        )
 
    # ── Ensemble Logic ────────────────────────────────────────────────────────
 
    @staticmethod
    def _ensemble_merge(
        primary: List[TextBlock], secondary: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Simple IoU-based deduplication: for each secondary block, if it
        overlaps significantly with a primary block, keep the higher-confidence
        one; otherwise append it (fills detection gaps).
        """
 
        def iou(a: BoundingBox, b: BoundingBox) -> float:
            ix1 = max(a.x1, b.x1)
            iy1 = max(a.y1, b.y1)
            ix2 = min(a.x2, b.x2)
            iy2 = min(a.y2, b.y2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = a.area + b.area - inter
            return inter / union if union else 0.0
 
        merged = list(primary)
        for sec in secondary:
            overlaps = [(i, iou(p.bbox, sec.bbox)) for i, p in enumerate(merged)]
            best_i, best_iou = max(overlaps, key=lambda x: x[1])
            if best_iou > 0.5:
                # Replace if secondary has higher confidence
                if sec.confidence > merged[best_i].confidence:
                    sec.engine = "ensemble"
                    merged[best_i] = sec
            else:
                sec.engine = "ensemble_gap_fill"
                merged.append(sec)
 
        return merged
 
    # ── Batch processing ──────────────────────────────────────────────────────
 
    def run_batch(
        self,
        image_paths: List[str],
        output_format: str = "text",
    ) -> List[OCRResult]:
        results = []
        for path in image_paths:
            try:
                r = self.run(path)
                results.append(r)
                if output_format == "json":
                    print(r.to_json())
                elif output_format == "markdown":
                    print(r.to_markdown())
                else:
                    print(f"\n{'─'*60}")
                    print(f"FILE : {path}")
                    print(f"CONF : {r.overall_confidence:.1%}  ({r.metadata['quality']})")
                    print(f"TIME : {r.processing_time_ms:.0f} ms")
                    print(f"TEXT :\n{r.raw_text}")
            except Exception as e:
                logger.error(f"Failed on {path}: {e}")
        return results
 
 
# ─── CLI ──────────────────────────────────────────────────────────────────────
 
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production OCR Pipeline — image → text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_pipeline.py photo.jpg
  python ocr_pipeline.py *.jpg --languages en hi --output json
  python ocr_pipeline.py label.png --ensemble --save-debug
  python ocr_pipeline.py scan.tiff --binarize --languages en
        """,
    )
    p.add_argument("images", nargs="*", help="Input image path(s). Supports glob patterns. If omitted, you will be prompted to enter them interactively.")
    p.add_argument("--languages", "-l", nargs="+", default=["en"],
                   help="OCR languages (EasyOCR codes). Default: en. E.g. en hi bn")
    p.add_argument("--gpu", action="store_true", help="Use GPU (if available).")
    p.add_argument("--ensemble", action="store_true",
                   help="Merge EasyOCR + Tesseract results.")
    p.add_argument("--confidence-threshold", type=float, default=0.60,
                   help="If EasyOCR confidence < threshold, fallback to Tesseract. Default: 0.60")
    p.add_argument("--binarize", action="store_true",
                   help="Apply adaptive binarization (good for low-contrast scans).")
    p.add_argument("--no-deskew", action="store_true", help="Skip deskew step.")
    p.add_argument("--no-denoise", action="store_true", help="Skip denoise step.")
    p.add_argument("--output", "-o", choices=["text", "json", "markdown"], default="text",
                   help="Output format. Default: text")
    p.add_argument("--output-dir", default="ocr_output",
                   help="Directory for outputs (JSON / debug images). Default: ocr_output")
    p.add_argument("--save-debug", action="store_true",
                   help="Save preprocessed image + annotated bounding boxes.")
    p.add_argument("--tesseract-lang", default="eng",
                   help="Tesseract language string. Default: eng. E.g. eng+hin")
    p.add_argument("--save-json", action="store_true",
                   help="Save each result as a JSON file in --output-dir.")
    return p
 
 
def main():
    parser = build_parser()
    args   = parser.parse_args()
 
    # Resolve glob patterns
    from glob import glob
    import shlex
 
    image_inputs = args.images
    if not image_inputs:
        try:
            user_input = input("Please enter the path to the image(s) (space-separated, supports drag-and-drop & wildcards): ").strip()
            if not user_input:
                logger.error("No image path entered. Exiting.")
                sys.exit(1)
            image_inputs = shlex.split(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)
 
    image_paths = []
    for pattern in image_inputs:
        pattern_expanded = os.path.expanduser(pattern)
        matches = glob(pattern_expanded)
        if matches:
            image_paths.extend(matches)
        else:
            image_paths.append(pattern_expanded)   # will raise FileNotFoundError gracefully
 
    preprocess_config = {
        "deskew":           not args.no_deskew,
        "denoise":          not args.no_denoise,
        "contrast_enhance": True,
        "shadow_remove":    True,
        "border_remove":    True,
        "binarize":         args.binarize,
        "upscale_threshold": 800,
        "upscale_factor":    2.0,
    }
 
    pipeline = OCRPipeline(
        languages=args.languages,
        gpu=args.gpu,
        preprocess_config=preprocess_config,
        confidence_threshold=args.confidence_threshold,
        ensemble=args.ensemble,
        tesseract_lang=args.tesseract_lang,
        save_debug_images=args.save_debug,
        output_dir=args.output_dir,
    )
 
    results = pipeline.run_batch(image_paths, output_format=args.output)
 
    if args.save_json:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            stem     = Path(r.image_path).stem
            out_path = out_dir / f"{stem}_ocr.json"
            out_path.write_text(r.to_json(), encoding="utf-8")
            logger.info(f"JSON saved: {out_path}")
 
 
# ─── Programmatic API (importable) ────────────────────────────────────────────
 
def ocr(
    image_path: str,
    languages: List[str] = None,
    gpu: bool = False,
    ensemble: bool = False,
    preprocess: bool = True,
) -> str:
    """
    One-liner API for quick integration::
 
        from ocr_pipeline import ocr
        text = ocr("packet.jpg", languages=["en", "hi"])
    """
    pre_cfg = {
        "deskew":           preprocess,
        "denoise":          preprocess,
        "contrast_enhance": preprocess,
        "shadow_remove":    preprocess,
        "border_remove":    preprocess,
        "binarize":         False,
    }
    pipe = OCRPipeline(
        languages=languages or ["en"],
        gpu=gpu,
        preprocess_config=pre_cfg,
        ensemble=ensemble,
    )
    result = pipe.run(image_path)
    return result.raw_text
 
 
def ocr_full(
    image_path: str,
    languages: List[str] = None,
    gpu: bool = False,
    ensemble: bool = False,
) -> OCRResult:
    """Same as ocr() but returns the full OCRResult dataclass."""
    pipe = OCRPipeline(languages=languages or ["en"], gpu=gpu, ensemble=ensemble)
    return pipe.run(image_path)
 
 
# ─── Entry-point ──────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    main()