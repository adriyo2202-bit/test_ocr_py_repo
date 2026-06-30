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
from PIL import Image, ExifTags

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

try:
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    try:
        from docling.datamodel.base_models import DocumentStream
    except ImportError:
        from docling_core.types.io import DocumentStream
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    warnings.warn("Docling not available. Install: pip install docling")

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
    x1: int; y1: int; x2: int; y2: int

    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def area(self) -> int: return self.width * self.height
    @property
    def center(self) -> Tuple[int, int]: return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: BoundingBox
    engine: str
    line_number: int = 0
    block_type: str = "text"
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
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        return f"# OCR Result\n**Image:** `{self.image_path}`\n**Confidence:** {self.overall_confidence:.1%}\n**Engine:** {self.engine_used}\n**Time:** {self.processing_time_ms:.0f} ms\n**Preprocessing:** {', '.join(self.preprocessing_applied) or 'none'}\n\n---\n\n## Extracted Text\n\n{self.raw_text}"


# ─── Image Preprocessor ───────────────────────────────────────────────────────

class ImagePreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.deskew            = cfg.get("deskew", True)
        self.denoise           = cfg.get("denoise", True)
        self.contrast_enhance  = cfg.get("contrast_enhance", True)
        self.binarize          = cfg.get("binarize", False)
        self.upscale_threshold = cfg.get("upscale_threshold", 1000)
        self.upscale_factor    = cfg.get("upscale_factor", 2.0)
        self.border_remove     = cfg.get("border_remove", True)
        self.shadow_remove     = cfg.get("shadow_remove", True)

    def process(self, img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        steps: List[str] = []
        img = self._fix_orientation(img, steps)
        img = self._upscale_if_small(img, steps)
        if self.shadow_remove: img = self._remove_shadows(img, steps)
        if self.denoise: img = self._denoise(img, steps)
        if self.contrast_enhance: img = self._enhance_contrast(img, steps)
        if self.deskew: img = self._deskew(img, steps)
        if self.border_remove: img = self._remove_borders(img, steps)
        if self.binarize: img = self._binarize(img, steps)
        return img, steps

    @staticmethod
    def _fix_orientation(img: np.ndarray, steps: List[str]) -> np.ndarray:
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
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            steps.append(f"upscale_{scale}x")
        return img

    @staticmethod
    def _remove_shadows(img: np.ndarray, steps: List[str]) -> np.ndarray:
        planes = []
        for plane in cv2.split(img):
            bg = cv2.medianBlur(cv2.dilate(plane, np.ones((7, 7), np.uint8)), 21)
            planes.append(cv2.normalize(255 - cv2.absdiff(plane, bg), None, 0, 255, cv2.NORM_MINMAX))
        steps.append("shadow_removal")
        return cv2.merge(planes)

    @staticmethod
    def _denoise(img: np.ndarray, steps: List[str]) -> np.ndarray:
        steps.append("nlm_denoise")
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    @staticmethod
    def _enhance_contrast(img: np.ndarray, steps: List[str]) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        steps.append("clahe_contrast")
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _deskew(img: np.ndarray, steps: List[str]) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0] if -45 < np.degrees(np.arctan2(y2 - y1, x2 - x1)) < 45]
            if angles:
                angle = float(np.median(angles))
                if abs(angle) >= 0.5:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    steps.append(f"deskew_{angle:.1f}deg")
        return img

    @staticmethod
    def _remove_borders(img: np.ndarray, steps: List[str]) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w * h > 0.5 * img.shape[0] * img.shape[1]:
                steps.append("border_remove")
                return img[y:y + h, x:x + w]
        return img

    @staticmethod
    def _binarize(img: np.ndarray, steps: List[str]) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        steps.append("adaptive_binarize")
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─── OCR Engines ──────────────────────────────────────────────────────────────

class EasyOCREngine:
    _reader_cache: Dict[str, "easyocr.Reader"] = {}

    def __init__(self, languages: List[str], gpu: bool = False):
        if not EASYOCR_AVAILABLE: raise RuntimeError("EasyOCR is not installed.")
        self.languages, self.gpu = languages, gpu
        key = "_".join(sorted(languages)) + f"_gpu{gpu}"
        if key not in self._reader_cache:
            logger.info("Initialising EasyOCR reader…")
            self._reader_cache[key] = easyocr.Reader(languages, gpu=gpu)
        self._reader = self._reader_cache[key]

    def read(self, img: np.ndarray) -> List[TextBlock]:
        results = self._reader.readtext(img, detail=1, paragraph=False, text_threshold=0.7, contrast_ths=0.1)
        blocks = []
        for i, (pts, text, conf) in enumerate(results):
            if not text.strip(): continue
            pts = np.array(pts, dtype=int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            blocks.append(TextBlock(text.strip(), float(conf), BoundingBox(int(x1), int(y1), int(x2), int(y2)), "easyocr", i))
        return blocks


class TesseractEngine:
    def __init__(self, lang: str = "eng", psm: int = 3):
        if not TESSERACT_AVAILABLE: raise RuntimeError("pytesseract is not installed.")
        self.lang, self.psm = lang, psm

    def read(self, img: np.ndarray) -> List[TextBlock]:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            data = pytesseract.image_to_data(pil_img, lang=self.lang, config=f"--oem 3 --psm {self.psm}", output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return []
        blocks = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 0: continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append(TextBlock(text, conf / 100.0, BoundingBox(x, y, x + w, y + h), "tesseract", data["line_num"][i]))
        return blocks


class DoclingEngine:
    def __init__(self, tesseract_langs: Optional[List[str]] = None, easyocr_langs: Optional[List[str]] = None):
        if not DOCLING_AVAILABLE: raise RuntimeError("Docling is not installed.")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        if TESSERACT_AVAILABLE:
            try:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions
                pipeline_options.ocr_options = TesseractCliOcrOptions(lang=tesseract_langs or ["eng"])
                logger.info(f"Docling using Tesseract CLI with languages {tesseract_langs or ['eng']}.")
            except Exception as e:
                logger.warning(f"Could not load Tesseract Cli options for Docling: {e}")
        else:
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                pipeline_options.ocr_options = EasyOcrOptions(lang=easyocr_langs or ["en"])
                logger.info(f"Docling using EasyOCR with languages {easyocr_langs or ['en']}.")
            except Exception as e:
                logger.warning(f"Could not load EasyOCR options for Docling: {e}")
        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def read(self, source: Any, is_stream: bool = False) -> Tuple[str, List[TextBlock]]:
        try:
            if is_stream:
                from io import BytesIO
                success, encoded = cv2.imencode(".png", source)
                if not success: raise ValueError("Failed to encode image to PNG.")
                doc_source = DocumentStream(name="preprocessed.png", stream=BytesIO(encoded.tobytes()))
            else:
                doc_source = Path(source)
            result = self._converter.convert(doc_source)
            doc = result.document
            markdown_text = doc.export_to_markdown()
            blocks = []
            page_sizes = {p_no: (p.size.width, p.size.height) for p_no, p in getattr(doc, "pages", {}).items() if getattr(p, "size", None)}
            for i, item in enumerate(doc.texts):
                text_content = item.text.strip()
                if not text_content: continue
                x1, y1, x2, y2, confidence = 0, 0, 0, 0, 0.90
                if getattr(item, "prov", None):
                    prov = item.prov[0]
                    bbox, page_no = getattr(prov, "bbox", None), getattr(prov, "page_no", None)
                    if bbox and page_no in page_sizes:
                        page_w, page_h = page_sizes[page_no]
                        try:
                            tl = bbox.to_top_left_origin(page_h)
                            x1, y1, x2, y2 = int(tl.l), int(tl.t), int(tl.r), int(tl.b)
                        except AttributeError:
                            if getattr(bbox, "coord_origin", "").endswith("BOTTOMLEFT"):
                                x1, y1, x2, y2 = int(bbox.l), int(page_h - bbox.t), int(bbox.r), int(page_h - bbox.b)
                            else:
                                x1, y1, x2, y2 = int(bbox.l), int(bbox.t), int(bbox.r), int(bbox.b)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                blocks.append(TextBlock(text_content, confidence, BoundingBox(x1, y1, x2, y2), "docling", i, getattr(item, "label", "text")))
            return markdown_text, blocks
        except Exception as e:
            logger.warning(f"Docling engine failed: {e}")
            return "", []


# ─── Layout Reconstructor ─────────────────────────────────────────────────────

class LayoutReconstructor:
    def __init__(self, line_y_tolerance: float = 0.5):
        self.line_y_tolerance = line_y_tolerance

    def reconstruct(self, blocks: List[TextBlock]) -> str:
        if not blocks: return ""
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.center[1])
        heights = [b.bbox.height for b in sorted_blocks if b.bbox.height > 0]
        tol = (float(np.median(heights)) if heights else 20.0) * self.line_y_tolerance
        rows = []
        curr_row = [sorted_blocks[0]]
        ref_y = sorted_blocks[0].bbox.center[1]
        for blk in sorted_blocks[1:]:
            if abs(blk.bbox.center[1] - ref_y) <= tol:
                curr_row.append(blk)
            else:
                rows.append(curr_row)
                curr_row, ref_y = [blk], blk.bbox.center[1]
        rows.append(curr_row)
        lines = []
        for r in rows:
            r.sort(key=lambda b: b.bbox.x1)
            lines.append(" ".join(b.text for b in r))
        return "\n".join(lines)


# ─── Post-Processor ───────────────────────────────────────────────────────────

class TextPostProcessor:
    def clean(self, text: str, is_markdown: bool = False) -> str:
        if not text: return text
        if is_markdown:
            lines = [line.rstrip() for line in text.splitlines()]
            res = []
            for l in lines:
                if not l:
                    if not res or res[-1] != "": res.append("")
                else:
                    res.append(l)
            return "\n".join(res).strip()
        lines = [" ".join(line.split()) for line in text.splitlines() if " ".join(line.split())]
        return "\n".join(l for l in lines if len(l.strip()) >= 1)


# ─── Confidence Scorer ────────────────────────────────────────────────────────

class ConfidenceScorer:
    @staticmethod
    def score(blocks: List[TextBlock]) -> float:
        if not blocks: return 0.0
        total_len = sum(len(b.text) for b in blocks)
        return sum(b.confidence * len(b.text) for b in blocks) / total_len if total_len else 0.0

    @staticmethod
    def quality_label(score: float) -> str:
        if score >= 0.90: return "excellent"
        if score >= 0.75: return "good"
        if score >= 0.55: return "fair"
        return "poor"


# ─── Main Pipeline ────────────────────────────────────────────────────────────

class OCRPipeline:
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
        use_docling: bool = True,
    ):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.ensemble = ensemble
        self.tesseract_lang = tesseract_lang
        self.save_debug_images = save_debug_images
        self.output_dir = Path(output_dir)
        self.use_docling = use_docling

        self.preprocessor = ImagePreprocessor(preprocess_config)
        self.layout = LayoutReconstructor()
        self.postproc = TextPostProcessor()
        self.scorer = ConfidenceScorer()

        self._easyocr_engine = None
        self._tesseract_engine = None
        self._docling_engine = None

        if self.save_debug_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def easyocr(self):
        if self._easyocr_engine is None:
            self._easyocr_engine = EasyOCREngine(self.languages, self.gpu)
        return self._easyocr_engine

    @property
    def tesseract(self):
        if self._tesseract_engine is None:
            self._tesseract_engine = TesseractEngine(self.tesseract_lang)
        return self._tesseract_engine

    @property
    def docling(self):
        if self._docling_engine is None:
            tess_langs = self.tesseract_lang.split("+") if self.tesseract_lang else ["eng"]
            self._docling_engine = DoclingEngine(tesseract_langs=tess_langs, easyocr_langs=self.languages)
        return self._docling_engine

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        ext = Path(path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None: return img
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _save_debug(self, img: np.ndarray, blocks: List[TextBlock], stem: str):
        vis = img.copy()
        for blk in blocks:
            b = blk.bbox
            color = (0, 255, 0) if blk.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            cv2.putText(vis, f"{blk.confidence:.2f}", (b.x1, max(b.y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(self.output_dir / f"{stem}_debug.jpg"), vis)
        logger.info(f"Debug image saved: {self.output_dir / f'{stem}_debug.jpg'}")

    def run(self, image_path: str) -> OCRResult:
        t0 = time.perf_counter()
        logger.info(f"Processing: {image_path}")
        img_raw = self.load_image(image_path)
        h, w = img_raw.shape[:2]
        logger.info(f"Image size: {w}×{h} px")

        img_proc, prep_steps = self.preprocessor.process(img_raw)
        logger.info(f"Preprocessing: {prep_steps}")

        if self.save_debug_images:
            stem = Path(image_path).stem
            cv2.imwrite(str(self.output_dir / f"{stem}_preprocessed.jpg"), img_proc)

        blocks: List[TextBlock] = []
        engine_used = "none"
        raw_text = ""

        if self.use_docling and DOCLING_AVAILABLE:
            try:
                raw_text, blocks = self.docling.read(img_proc if prep_steps else image_path, is_stream=bool(prep_steps))
                engine_used = "docling"
                logger.info(f"Docling: {len(blocks)} blocks, confidence={self.scorer.score(blocks):.1%}")
            except Exception as e:
                logger.warning(f"Docling failed: {e}")

        if not blocks and EASYOCR_AVAILABLE:
            try:
                blocks = self.easyocr.read(img_proc)
                engine_used = "easyocr"
                logger.info(f"EasyOCR: {len(blocks)} blocks, confidence={self.scorer.score(blocks):.1%}")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")

        if TESSERACT_AVAILABLE:
            primary_conf = self.scorer.score(blocks)
            if primary_conf < self.confidence_threshold or not blocks or self.ensemble:
                try:
                    tess_blocks = self.tesseract.read(img_proc)
                    tess_conf = self.scorer.score(tess_blocks)
                    logger.info(f"Tesseract: {len(tess_blocks)} blocks, confidence={tess_conf:.1%}")
                    if self.ensemble and blocks:
                        blocks = self._ensemble_merge(blocks, tess_blocks)
                        engine_used = "ensemble"
                    elif tess_conf > primary_conf:
                        blocks = tess_blocks
                        engine_used = "tesseract"
                except Exception as e:
                    logger.warning(f"Tesseract fallback failed: {e}")

        if not blocks:
            logger.warning("No text detected by any engine.")

        if self.save_debug_images and blocks:
            self._save_debug(img_proc, blocks, Path(image_path).stem)

        if engine_used != "docling":
            raw_text = self.layout.reconstruct(blocks)

        clean_text = self.postproc.clean(raw_text, is_markdown=(engine_used == "docling"))
        overall_conf = self.scorer.score(blocks)
        quality = self.scorer.quality_label(overall_conf)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Done in {elapsed_ms:.0f} ms | engine={engine_used} | confidence={overall_conf:.1%} ({quality})")

        return OCRResult(
            raw_text=clean_text,
            structured_blocks=blocks,
            overall_confidence=overall_conf,
            processing_time_ms=elapsed_ms,
            image_path=image_path,
            image_size=(w, h),
            preprocessing_applied=prep_steps,
            engine_used=engine_used,
            metadata={"quality": quality, "block_count": len(blocks), "languages": self.languages, "gpu": self.gpu},
        )

    @staticmethod
    def _ensemble_merge(primary: List[TextBlock], secondary: List[TextBlock]) -> List[TextBlock]:
        def iou(a: BoundingBox, b: BoundingBox) -> float:
            ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
            ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
            if ix2 <= ix1 or iy2 <= iy1: return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = a.area + b.area - inter
            return inter / union if union else 0.0

        merged = list(primary)
        for sec in secondary:
            overlaps = [(i, iou(p.bbox, sec.bbox)) for i, p in enumerate(merged)]
            if overlaps:
                best_i, best_iou = max(overlaps, key=lambda x: x[1])
                if best_iou > 0.5:
                    if sec.confidence > merged[best_i].confidence:
                        sec.engine = "ensemble"
                        merged[best_i] = sec
                    continue
            sec.engine = "ensemble_gap_fill"
            merged.append(sec)
        return merged

    def run_batch(self, image_paths: List[str], output_format: str = "text") -> List[OCRResult]:
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
                    print(f"\n{'─'*60}\nFILE : {path}\nCONF : {r.overall_confidence:.1%} ({r.metadata['quality']})\nTIME : {r.processing_time_ms:.0f} ms\nTEXT :\n{r.raw_text}")
            except Exception as e:
                logger.error(f"Failed on {path}: {e}")
        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Production OCR Pipeline — image → text")
    p.add_argument("images", nargs="*", help="Input image path(s). Supports glob patterns.")
    p.add_argument("--languages", "-l", nargs="+", default=["en"], help="OCR languages (EasyOCR). Default: en")
    p.add_argument("--gpu", action="store_true", help="Use GPU (if available).")
    p.add_argument("--ensemble", action="store_true", help="Merge EasyOCR + Tesseract.")
    p.add_argument("--docling", action="store_true", default=True, help="Use Docling as primary (default).")
    p.add_argument("--no-docling", action="store_false", dest="docling", help="Disable Docling (use EasyOCR).")
    p.add_argument("--confidence-threshold", type=float, default=0.60, help="Fallback threshold. Default: 0.60")
    p.add_argument("--binarize", action="store_true", help="Binarize image.")
    p.add_argument("--no-deskew", action="store_true", help="Skip deskew.")
    p.add_argument("--no-denoise", action="store_true", help="Skip denoise.")
    p.add_argument("--output", "-o", choices=["text", "json", "markdown"], default="text", help="Output format. Default: text")
    p.add_argument("--output-dir", default="ocr_output", help="Output directory. Default: ocr_output")
    p.add_argument("--save-debug", action="store_true", help="Save debug images.")
    p.add_argument("--tesseract-lang", default="eng", help="Tesseract lang. Default: eng")
    p.add_argument("--save-json", action="store_true", help="Save JSON files.")
    return p


def main():
    args = build_parser().parse_args()
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
            sys.exit(1)

    image_paths = []
    for pattern in image_inputs:
        matches = glob(os.path.expanduser(pattern))
        image_paths.extend(matches if matches else [os.path.expanduser(pattern)])

    pipeline = OCRPipeline(
        languages=args.languages,
        gpu=args.gpu,
        preprocess_config={
            "deskew": not args.no_deskew,
            "denoise": not args.no_denoise,
            "contrast_enhance": True,
            "shadow_remove": True,
            "border_remove": True,
            "binarize": args.binarize,
            "upscale_threshold": 800,
            "upscale_factor": 2.0,
        },
        confidence_threshold=args.confidence_threshold,
        ensemble=args.ensemble,
        tesseract_lang=args.tesseract_lang,
        save_debug_images=args.save_debug,
        output_dir=args.output_dir,
        use_docling=args.docling,
    )
    results = pipeline.run_batch(image_paths, output_format=args.output)

    if args.save_json:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            (out_dir / f"{Path(r.image_path).stem}_ocr.json").write_text(r.to_json(), encoding="utf-8")


# ─── Programmatic API (importable) ────────────────────────────────────────────

def ocr(
    image_path: str,
    languages: List[str] = None,
    gpu: bool = False,
    ensemble: bool = False,
    preprocess: bool = True,
    use_docling: bool = True,
) -> str:
    pipe = OCRPipeline(
        languages=languages or ["en"],
        gpu=gpu,
        preprocess_config={k: preprocess for k in ["deskew", "denoise", "contrast_enhance", "shadow_remove", "border_remove"]},
        ensemble=ensemble,
        use_docling=use_docling,
    )
    return pipe.run(image_path).raw_text


def ocr_full(
    image_path: str,
    languages: List[str] = None,
    gpu: bool = False,
    ensemble: bool = False,
    use_docling: bool = True,
) -> OCRResult:
    pipe = OCRPipeline(languages=languages or ["en"], gpu=gpu, ensemble=ensemble, use_docling=use_docling)
    return pipe.run(image_path)


# ─── Entry-point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()