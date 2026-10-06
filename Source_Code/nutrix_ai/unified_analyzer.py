import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.ocr_new import ocr

def get_ingredient_safety_rules(raw_text: str) -> str:
    """Queries the SQLite RAG DB for rules related to words found in the OCR text."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'nutrix_rag.db')
    if not os.path.exists(db_path):
        return "RAG database not found."
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Simple keyword extraction (words longer than 4 chars)
        words = [w.strip(".,;:()") for w in raw_text.split() if len(w) > 4]
        if not words:
            return "No valid keywords for RAG."
            
        rules = []
        for word in set(words[:20]): # Limit to first 20 unique words to avoid massive queries
            cursor.execute("SELECT ingredient_name, fssai_status, eu_status, primary_health_risks FROM ingredient_safety_rules WHERE LOWER(ingredient_name) LIKE ?", (f"%{word.lower()}%",))
            for row in cursor.fetchall():
                rules.append(f"- {row['ingredient_name']}: EU Status: {row['eu_status']}, FSSAI: {row['fssai_status']}. Risks: {row['primary_health_risks']}")
                
        conn.close()
        # Deduplicate and return
        unique_rules = list(set(rules))
        if not unique_rules:
             return "No matching RAG safety rules found in database."
        return "\n".join(unique_rules[:15]) # Limit to 15 rules max for context
    except Exception as e:
        return f"RAG error: {e}"

def run_label_analysis(image_path: str) -> dict:
    """
    1. Extracts clean OCR text via YOLOv8 + Tesseract
    2. Queries RAG database for safety rules matching the extracted text
    3. Prompts the Local Llama 3.1 8B text model to format the final JSON structure
    """
    
    print(f"Running YOLOv8 OCR on {image_path}...")
    try:
        raw_text = ocr(image_path)
    except Exception as e:
        print(f"OCR failed, falling back to empty string: {e}")
        raw_text = ""
        
    print(f"Extracted Base Text: {raw_text[:150]}...")
    
    print("Fetching RAG context from SQLite...")
    rag_rules = get_ingredient_safety_rules(raw_text)
    
    # Local Llama unified prompt
    prompt = f"""You are 'Healthy Food AI', an expert nutrition and food safety analyst. 
I have scanned a nutrition label. Here is the raw OCR text extracted from the image:

<raw_ocr>
{raw_text}
</raw_ocr>

Here are the strict safety rules from our database that apply to some of these ingredients:
<database_rules>
{rag_rules}
</database_rules>

Analyze the OCR text carefully. Correct any OCR mistakes. Then, extract the ingredients and analyze them for health, safety, and EU standards. Also suggest healthy alternatives.
If an ingredient is listed in the <database_rules>, you MUST use those exact verdicts and risks.

IMPORTANT: The entire JSON response MUST be written strictly in ENGLISH. The ONLY exception is the `name_bn` field which should be in Bengali. All explanations, summaries, and reasons MUST be in English.

You MUST respond with a JSON object exactly matching this structure. The values for product_ingredients_raw and summary_bn_en MUST be flat strings, NOT nested objects. Do NOT include markdown tags, just the raw JSON:
{{
  "product_ingredients_raw": "A single string containing the corrected, full text of the ingredients list.",
  "ingredients": [
    {{
      "name": "Ingredient Name",
      "name_bn": "Bengali translation of ingredient (or empty string)",
      "safety_verdict": "safe" | "caution" | "unsafe",
      "safety_reason": "Explanation of why it is safe/unsafe.",
      "eu_status": "permitted" | "banned" | "restricted",
      "eu_e_number": "E-number if applicable, else null",
      "eu_notes": "Any EU regulations or notes",
      "long_term_risk": true | false,
      "long_term_risk_detail": "Details of long-term risk, or empty string",
      "risk_severity": "none" | "low" | "moderate" | "high"
    }}
  ],
  "overall_safety_score": <integer from 0 to 100>,
  "warnings_detected": ["List of any allergens or warnings"],
  "allergen_flags": ["List of allergen names"],
  "summary_bn_en": "A single string containing a summary of the product's safety. Include a section for 'Healthy Alternatives' suggesting better options."
}}
"""

    url = "http://127.0.0.1:8081/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    print("Calling Local Llama Server (Unified Analysis)...")
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            result_text = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
        
        parsed_json = json.loads(result_text)
        
        # Enforce string types for UI compatibility
        if isinstance(parsed_json.get("product_ingredients_raw"), dict):
            parsed_json["product_ingredients_raw"] = json.dumps(parsed_json["product_ingredients_raw"], indent=2)
        if isinstance(parsed_json.get("summary_bn_en"), dict):
            # Try to format it nicely if it separated by language
            summary = parsed_json["summary_bn_en"]
            if "en" in summary and "bn" in summary:
                parsed_json["summary_bn_en"] = f"{json.dumps(summary['en'], indent=2)}\n\n{json.dumps(summary['bn'], indent=2)}"
            else:
                parsed_json["summary_bn_en"] = json.dumps(summary, indent=2)
                
        print("Unified Analysis Complete!")
        return parsed_json
        
    except Exception as e:
        print(f"Error calling Local Llama API: {e}")
        # Fallback to mock data if API fails
        return {
          "product_ingredients_raw": "Local Inference Failed",
          "ingredients": [
            {
              "name": "Local Server Error",
              "name_bn": "",
              "safety_verdict": "unknown",
              "safety_reason": f"Could not connect to local Llama server on port 8081. Error: {e}",
              "eu_status": "unknown",
              "eu_e_number": None,
              "eu_notes": "",
              "long_term_risk": False,
              "long_term_risk_detail": "",
              "risk_severity": "none"
            }
          ],
          "overall_safety_score": 0,
          "warnings_detected": ["Server Error"],
          "allergen_flags": [],
          "summary_bn_en": "Please ensure the local Llama server is running."
        }
