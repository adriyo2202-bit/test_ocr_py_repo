import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
def check_dependencies():
    """Checks which OCR engines are available in the current environment."""
    easyocr_available = False
    pytesseract_available = False
    try:
        import easyocr
        easyocr_available = True
    except ImportError:
        pass
    try:
        import pytesseract
        # Try to run a quick version check to see if the tesseract command is in PATH
        pytesseract.get_tesseract_version()
        pytesseract_available = True
    except (ImportError, pytesseract.TesseractNotFoundError):
        pass
    return easyocr_available, pytesseract_available
def run_easyocr(image_path, lang='en', visualize=False):
    """Performs OCR using EasyOCR and returns the text and bounding boxes."""
    try:
        import easyocr
        import numpy as np
    except ImportError:
        print("Error: easyocr or numpy is not installed.", file=sys.stderr)
        return None, None
    print(f"[*] Initializing EasyOCR reader for language: '{lang}'...")
    # EasyOCR might download models on first run
    reader = easyocr.Reader([lang])
    
    print("[*] Running text detection and recognition...")
    # Read text. By default, easyocr takes a file path, numpy array, or PIL Image.
    results = reader.readtext(image_path)
    
    extracted_text = []
    boxes = []
    
    for bbox, text, confidence in results:
        extracted_text.append(text)
        # bbox is a list of 4 points: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        boxes.append({
            'box': bbox,
            'text': text,
            'confidence': confidence
        })
        
    full_text = "\n".join(extracted_text)
    return full_text, boxes
def run_pytesseract(image_path, lang='eng', visualize=False):
    """Performs OCR using PyTesseract."""
    try:
        import pytesseract
    except ImportError:
        print("Error: pytesseract is not installed.", file=sys.stderr)
        return None, None
    print("[*] Running Tesseract OCR...")
    try:
        img = Image.open(image_path)
        # Get raw text
        full_text = pytesseract.image_to_string(img, lang=lang)
        
        # Get structured data for bounding boxes if visualization is requested
        boxes = []
        if visualize:
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                # Filter out empty text detections or low confidence
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    # Convert to standard 4-point bounding box format: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    boxes.append({
                        'box': bbox,
                        'text': data['text'][i],
                        'confidence': float(data['conf'][i]) / 100.0
                    })
        return full_text, boxes
    except Exception as e:
        print(f"Error executing Tesseract OCR: {e}", file=sys.stderr)
        return None, None
def draw_visualizations(image_path, boxes, output_path):
    """Draws bounding boxes and labels on the image and saves it."""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Error opening image for visualization: {e}", file=sys.stderr)
        return
    print(f"[*] Visualizing {len(boxes)} text detections...")
    
    # Try to load a default font, or fallback to default
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None
    for item in boxes:
        box = item['box']
        text = item['text']
        conf = item['confidence']
        
        # Flatten the box to (x0, y0, x1, y1) for PIL's rectangle drawing if it has 4 points
        # EasyOCR returns [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        # Tesseract returns [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        
        # Draw label background
        label = f"{text} ({conf:.2f})"
        # Draw text at the top-left of the box
        draw.text((x0, max(0, y0 - 15)), label, fill="blue", font=font)
    try:
        img.save(output_path)
        print(f"[+] Visualized output saved to: {output_path}")
    except Exception as e:
        print(f"Error saving visualized image: {e}", file=sys.stderr)
def main():
    parser = argparse.ArgumentParser(description="Python-based OCR Command Line Tool")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file")
    parser.add_argument("-e", "--engine", choices=["auto", "easyocr", "tesseract"], default="auto",
                        help="OCR Engine to use (default: auto)")
    parser.add_argument("-o", "--output", help="Path to save the extracted text output file")
    parser.add_argument("-l", "--lang", default="en", help="Language code (e.g. 'en', 'es', 'fr', etc.)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Generate a copy of the image with bounding boxes drawn")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        print(f"Error: Input image file '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)
    easyocr_avail, tesseract_avail = check_dependencies()
    print("[*] Checking OCR engines available in environment...")
    print(f"    - EasyOCR available: {easyocr_avail}")
    print(f"    - Tesseract (pytesseract) available: {tesseract_avail}")
    selected_engine = args.engine
    if selected_engine == "auto":
        if easyocr_avail:
            selected_engine = "easyocr"
        elif tesseract_avail:
            selected_engine = "tesseract"
        else:
            print("Error: No OCR engine is available. Please run installation step first.", file=sys.stderr)
            sys.exit(1)
    print(f"[*] Using OCR engine: '{selected_engine}'")
    text = None
    boxes = None
    if selected_engine == "easyocr":
        # Lang mapping: EasyOCR uses 'en', Tesseract uses 'eng'
        lang_code = args.lang
        text, boxes = run_easyocr(args.image, lang=lang_code, visualize=args.visualize)
    elif selected_engine == "tesseract":
        # Lang mapping: Tesseract typically uses 3 letter codes for some languages, mapping default 'en' -> 'eng'
        lang_code = "eng" if args.lang == "en" else args.lang
        text, boxes = run_pytesseract(args.image, lang=lang_code, visualize=args.visualize)
    if text is None:
        print("Error: Text extraction failed.", file=sys.stderr)
        sys.exit(1)
    print("\n--- EXTRACTED TEXT ---")
    print(text)
    print("----------------------\n")
    # Save to output file if requested
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[+] Extracted text saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output file: {e}", file=sys.stderr)
    # Save visualization if requested
    if args.visualize and boxes:
        base, ext = os.path.splitext(args.image)
        vis_output_path = f"{base}_detected{ext}"
        draw_visualizations(args.image, boxes, vis_output_path)
if __name__ == "_main_":
    main()