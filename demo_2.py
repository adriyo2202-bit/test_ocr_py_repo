import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont


def check_dependencies():
    """Checks which OCR engines are available."""
    easyocr_available = False
    pytesseract_available = False

    try:
        import easyocr
        easyocr_available = True
    except ImportError:
        pass

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        pytesseract_available = True
    except Exception:
        pass

    return easyocr_available, pytesseract_available


def run_easyocr(image_path, lang="en"):
    """Run OCR using EasyOCR."""
    try:
        import easyocr
    except ImportError:
        print("Error: EasyOCR is not installed.", file=sys.stderr)
        return None, None

    print(f"[*] Initializing EasyOCR ({lang})...")
    reader = easyocr.Reader([lang])

    print("[*] Running OCR...")
    results = reader.readtext(image_path)

    extracted_text = []
    boxes = []

    for bbox, text, confidence in results:
        extracted_text.append(text)
        boxes.append({
            "box": bbox,
            "text": text,
            "confidence": confidence
        })

    return "\n".join(extracted_text), boxes


def run_pytesseract(image_path, lang="eng"):
    """Run OCR using PyTesseract."""
    try:
        import pytesseract
    except ImportError:
        print("Error: pytesseract is not installed.", file=sys.stderr)
        return None, None

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        return text, []
    except Exception as e:
        print(e)
        return None, None


def draw_visualizations(image_path, boxes, output_path):
    """Draw OCR bounding boxes on image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    for item in boxes:
        box = item["box"]
        text = item["text"]
        conf = item["confidence"]

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text(
            (x0, max(0, y0 - 15)),
            f"{text} ({conf:.2f})",
            fill="blue",
            font=font
        )

    img.save(output_path)
    print(f"[+] Visualization saved as {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OCR Tool")

    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "-e",
        "--engine",
        default="auto",
        choices=["auto", "easyocr", "tesseract"]
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Save extracted text to file"
    )

    parser.add_argument(
        "-l",
        "--lang",
        default="en"
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Image not found.")
        sys.exit(1)

    easyocr_avail, tesseract_avail = check_dependencies()

    print("[*] Available OCR Engines")
    print("EasyOCR:", easyocr_avail)
    print("Tesseract:", tesseract_avail)

    engine = args.engine

    if engine == "auto":
        if easyocr_avail:
            engine = "easyocr"
        elif tesseract_avail:
            engine = "tesseract"
        else:
            print("No OCR engine installed.")
            sys.exit(1)

    print(f"[*] Using {engine}")

    if engine == "easyocr":
        text, boxes = run_easyocr(args.image, args.lang)
    else:
        lang = "eng" if args.lang == "en" else args.lang
        text, boxes = run_pytesseract(args.image, lang)

    if text is None:
        print("OCR failed.")
        sys.exit(1)

    print("\n========== EXTRACTED TEXT ==========\n")
    print(text)
    print("\n====================================\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[+] Text saved to {args.output}")

    if args.visualize and boxes:
        base, ext = os.path.splitext(args.image)
        draw_visualizations(
            args.image,
            boxes,
            f"{base}_detected{ext}"
        )


if __name__ == "__main__":
    main()