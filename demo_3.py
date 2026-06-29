import os
import easyocr

def main():
    # Ask the user for the image path
    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("❌ Image not found!")
        return

    print("Loading EasyOCR...")
    reader = easyocr.Reader(['en'])

    print("Running OCR...")
    results = reader.readtext(image_path)

    print("\n========== EXTRACTED TEXT ==========\n")

    if len(results) == 0:
        print("No text detected.")
    else:
        for result in results:
            print(result[1])

    print("\n====================================")


if __name__ == "__main__":
    main()