#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from ingre_demo_1 import NutriScanAnalyzer

def main():
    workspace_dir = Path(__file__).parent.resolve()
    
    # Prompt the user for the input method
    print("How would you like to provide the input?")
    print("  [1] Provide a file path (OCR text file)")
    print("  [2] Paste/type the OCR text directly into the console")
    choice = input("Enter option (1 or 2, default is 1): ").strip()

    ocr_text = ""
    if choice == "2":
        print("\nPlease paste/type your OCR text below.")
        print("When you are finished, press Ctrl-D (on Mac/Linux) or Ctrl-Z (on Windows) then press Enter:")
        ocr_text = sys.stdin.read()
    else:
        input_path = input("Enter the path to the OCR text file (e.g. ocr_output.txt): ").strip()
        while not input_path:
             input_path = input("Path cannot be empty. Please enter the path: ").strip()
             
        ocr_file = Path(input_path)
        if not ocr_file.is_file():
            print(f"[-] File not found or is not a file: {ocr_file}", file=sys.stderr)
            sys.exit(1)

        print("[*] Reading OCR text output...")
        ocr_text = ocr_file.read_text(encoding="utf-8")

    report_file = workspace_dir / "safety_report.md"

    print("[*] Initializing NutriScanAnalyzer (Mistral AI only)...")
    # Initialize. By default, if no API keys are found in .env, it will use the expert Mock mode.
    analyzer = NutriScanAnalyzer()

    print("[*] Running safety analysis...")
    try:
        result = analyzer.analyze_from_ocr_text(ocr_text)
    except Exception as e:
        print(f"[-] Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n========== SAFETY ANALYSIS REPORT ==========\n")
    report_md = result.to_markdown()
    print(report_md)
    print("\n============================================\n")

    # Save the report to safety_report.md
    report_file.write_text(report_md, encoding="utf-8")
    print(f"[+] Safety report successfully saved to: {report_file}")

if __name__ == "__main__":
    main()
