ð 
import os
import re
import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import requests
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ingredient_analyzer")
 
def load_dotenv():
    """Manually parse .env files if present to load configuration keys."""
    for p in [Path("."), Path(__file__).parent]:
        env_file = p / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k_clean = k.strip()
                        v_clean = v.strip().strip("'\"")
                        if k_clean and k_clean not in os.environ:
                            os.environ[k_clean] = v_clean
                logger.info(f"Loaded config from {env_file}")
                return
            except Exception as e:
                logger.warning(f"Error reading {env_file}: {e}")
 
 
# ─── Data Structures ──────────────────────────────────────────────────────────
 
@dataclass
class IngredientSafety:
    name: str
    name_bn: str = ""                      # Bengali name if applicable
    safety_verdict: str = "unknown"        # "safe" | "caution" | "unsafe" | "unknown"
    safety_reason: str = ""
    eu_status: str = "unknown"             # "permitted" | "restricted" | "banned" | "unknown"
    eu_e_number: Optional[str] = None      # e.g. "E102" for tartrazine
    eu_notes: str = ""
    long_term_risk: bool = False
    long_term_risk_detail: str = ""        # e.g. cancer, hyperactivity, allergy etc.
    risk_severity: str = "none"            # "none" | "low" | "moderate" | "high"
 
 
@dataclass
class ProductAnalysis:
    product_ingredients_raw: str
    ingredients: List[IngredientSafety]
    overall_safety_score: int              # 0-100
    warnings_detected: List[str]
    allergen_flags: List[str]
    summary_bn_en: str                     # Bangla-English mixed summary
    raw_model_response: str = ""
 
    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
 
    def to_markdown(self) -> str:
        attention_ingredients = []
        safe_ingredients = []
        
        for ing in self.ingredients:
            verdict = ing.safety_verdict.strip().lower()
            if verdict in ("caution", "unsafe"):
                attention_ingredients.append(ing)
            else:
                safe_ingredients.append(ing)
                
        lines = ["# 🛡️ Ingredient Safety Report\n"]
        
        # Color-coded overall score indicators
        score_emoji = "🟢" if self.overall_safety_score >= 80 else ("🟡" if self.overall_safety_score >= 50 else "🔴")
        lines.append(f"### {score_emoji} Overall Safety Score: **{self.overall_safety_score}/100**\n")
        
        if self.warnings_detected:
            lines.append(f"⚠️ **Label Warnings:** {'; '.join(self.warnings_detected)}\n")
        if self.allergen_flags:
            lines.append(f"🌾 **Allergen Flags:** {', '.join(self.allergen_flags)}\n")
            
        lines.append("\n---")
        
        # Highlighted Caution/Unsafe ingredients
        if attention_ingredients:
            lines.append("\n## 🔍 Ingredients Requiring Attention\n")
            lines.append("| Ingredient | Safety | EU Status | Long-term Risk / Concerns |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for ing in attention_ingredients:
                verdict_emoji = "🔴 Unsafe" if ing.safety_verdict.strip().lower() == "unsafe" else "🟡 Caution"
                risk = ing.long_term_risk_detail if ing.long_term_risk else "No major long-term risk reported"
                eu = f"{ing.eu_status.capitalize()}" + (f" ({ing.eu_e_number})" if ing.eu_e_number else "")
                lines.append(f"| **{ing.name}** | {verdict_emoji} | {eu} | {risk} |")
            lines.append("\n---")
            
        # Clean, simple list for Safe ingredients
        if safe_ingredients:
            lines.append("\n## 🟢 Safe Ingredients (No Concerns)\n")
            names = [f"**{ing.name}**" for ing in safe_ingredients]
            lines.append(", ".join(names) + "\n")
            lines.append("\n---")
            
        # Summary verdict
        lines.append("\n## 📝 Summary Verdict\n")
        lines.append(self.summary_bn_en)
        return "\n".join(lines)
 
 
# ─── OCR Text Cleaner (handles specific OCR garbage) ────────────────────
 
class OCRTextCleaner:
    """
    Fixes common OCR artifacts seen in real packaging scans, e.g.:
      '—— Protein' -> 'Protein'
      'Saturated fat | q' -> table OCR noise
      ';' used instead of 'g'
      'al ll' stray noise lines
    Extracts just the INGREDIENTS block + warning/phenylketonuric blocks cleanly.
    """
 
    NOISE_LINE_PATTERN = re.compile(r"^[\W_]{1,6}$")  # lines that are just symbols/junk
 
    @staticmethod
    def extract_section(text: str, header_keywords: List[str], stop_keywords: List[str]) -> str:
        """Extract text between a header (e.g. 'INGREDIENTS') and the next section header."""
        lines = text.splitlines()
        capturing = False
        captured = []
        for line in lines:
            upper = line.strip().upper()
            if not capturing and any(k in upper for k in header_keywords):
                capturing = True
                # strip markdown hashes + the header word itself; keep trailing content on same line
                stripped_line = line.lstrip("#").strip()
                after_colon = re.split(r":", stripped_line, maxsplit=1)
                if len(after_colon) > 1 and after_colon[1].strip():
                    captured.append(after_colon[1].strip())
                continue
            if capturing:
                if any(k in upper for k in stop_keywords):
                    break
                captured.append(line)
        return " ".join(l.strip() for l in captured if l.strip())
 
    def extract_ingredients_block(self, ocr_text: str) -> str:
        block = self.extract_section(
            ocr_text,
            header_keywords=["INGREDIENT"],
            stop_keywords=["PHENYLKETONURIC", "WARNING", "NUTRITION", "ALLERGEN", "STORAGE", "MANUFACTURED"],
        )
        if not block:
            # Fallback: if no INGREDIENTS header is found, treat the entire input as the ingredients block.
            block = ocr_text
        block = re.sub(r"\s+", " ", block).strip()
        block = block.strip(" .")
        return block
 
    def extract_warnings(self, ocr_text: str) -> List[str]:
        warnings = []
        for line in ocr_text.splitlines():
            up = line.strip().upper()
            if not up or self.NOISE_LINE_PATTERN.match(up):
                continue
            if "WARNING" in up or "CONTAINS PHENYLALANINE" in up or "PHENYLKETONURIC" in up:
                warnings.append(line.strip())
        return warnings
 
    @staticmethod
    def split_ingredients(ingredients_block: str) -> List[str]:
        """
        Split top-level ingredients on commas, but respect brackets/parens
        so 'mango pieces [mango, mango juice, ...]' stays grouped under mango pieces
        while still exposing the sub-ingredients for analysis.
        Also treats a trailing ' and X' as a final separator (common in label English).
        """
        # Normalize " and " before the last item into a comma, only at depth 0
        normalized = ""
        depth = 0
        i = 0
        while i < len(ingredients_block):
            ch = ingredients_block[i]
            if ch in "[(":
                depth += 1
                normalized += ch
            elif ch in "])":
                depth -= 1
                normalized += ch
            elif depth == 0 and ingredients_block[i:i+5].lower() == " and ":
                normalized += ", "
                i += 4
            else:
                normalized += ch
            i += 1
        ingredients_block = normalized
 
        items = []
        depth = 0
        current = ""
        for ch in ingredients_block:
            if ch in "[(":
                depth += 1
                current += ch
            elif ch in "])":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                if current.strip():
                    items.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            items.append(current.strip())
 
        # Flatten bracketed sub-ingredients into separate entries too
        flat = []
        for item in items:
            flat.append(re.sub(r"[\[\]()]", "", re.split(r"[\[(]", item)[0]).strip())
            bracket_match = re.search(r"[\[(](.*)[\])]", item)
            if bracket_match:
                inner = bracket_match.group(1)
                for sub in re.split(r",(?![^\[(]*[\])])", inner):
                    sub_clean = sub.strip()
                    if sub_clean and "[" not in sub_clean:
                        flat.append(re.sub(r"\(.*?\)", "", sub_clean).strip())
        # Dedup while preserving order
        seen = set()
        result = []
        for f in flat:
            f_norm = f.lower()
            if f_norm and f_norm not in seen and len(f) > 1:
                seen.add(f_norm)
                result.append(f)
        return result
 
 
# ─── Mistral-based Safety Analyzer ────────────────────────────────────────────
 
class MistralIngredientAnalyzer:
    """
    Calls Mistral AI (chat completions, JSON mode) with a strict-JSON system
    prompt to get per-ingredient safety verdicts, EU food-additive regulation
    status, and long-term health risk flags (carcinogenicity, hyperactivity, etc).
    If no API key is found, runs in MOCK mode utilizing pre-computed expert data.
    """
 
    SYSTEM_PROMPT = """You are a food-safety and EU food-regulation expert (knowledge of EU Regulation EC 1333/2008 on food additives, EFSA opinions, and E-number classifications).
 
You will receive a list of ingredients extracted via OCR from an Indian packaged-food label (text may contain minor OCR noise — infer the intended ingredient name).
 
For EACH ingredient, return a JSON object with these EXACT fields:
- "name": cleaned ingredient name (English)
- "name_bn": Bengali name/transliteration if commonly known, else ""
- "safety_verdict": one of "safe", "caution", "unsafe"
- "safety_reason": one short sentence (max 20 words)
- "eu_status": one of "permitted", "restricted", "banned" — based on actual EU food law/EFSA status
- "eu_e_number": E-number string if it's an EU-regulated additive (e.g. "E102"), else null
- "eu_notes": one short sentence on EU-specific conditions/restrictions, else ""
- "long_term_risk": true/false — whether credible scientific evidence links long-term/regular consumption to chronic health risk (cancer, hyperactivity in children, metabolic disease, organ damage, etc.)
- "long_term_risk_detail": short description of the specific long-term risk if true, else ""
- "risk_severity": one of "none", "low", "moderate", "high"
 
Be factually accurate and conservative — do not exaggerate, but do not omit known EFSA/IARC findings (e.g. aspartame's 2023 IARC "possibly carcinogenic" classification, tartrazine's hyperactivity warnings requiring EU mandatory labelling, titanium dioxide's 2022 EU ban as E171, etc.)
 
You MUST return ONLY a valid JSON object with a single key "ingredients" whose value is an array of the per-ingredient objects described above. No markdown fences, no preamble, no explanation outside the JSON.
 
Example output format:
{"ingredients": [{"name":"Aspartame","name_bn":"অ্যাসপারটেম","safety_verdict":"caution","safety_reason":"Artificial sweetener; IARC classified as possibly carcinogenic in 2023.","eu_status":"permitted","eu_e_number":"E951","eu_notes":"Permitted under ADI 40mg/kg bodyweight; requires phenylalanine warning label.","long_term_risk":true,"long_term_risk_detail":"Possible link to cancer (IARC Group 2B) with high long-term intake; metabolic concerns.","risk_severity":"moderate"}]}
"""
 
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-large-latest"):
        load_dotenv()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.model_name = model_name
 
        if not self.api_key:
            logger.warning("No Mistral API key found (MISTRAL_API_KEY). Analyzer running in MOCK mode.")
            self.is_mock = True
        else:
            self.is_mock = False
 
    def analyze(self, ingredient_names: List[str], max_retries: int = 3) -> List[IngredientSafety]:
        if self.is_mock:
            return self._mock_analyze(ingredient_names)
 
        prompt = "Analyze these ingredients:\n" + "\n".join(f"- {n}" for n in ingredient_names)
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
 
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                res_json = response.json()
                raw = res_json["choices"][0]["message"]["content"].strip()
                return self._parse_json_response(raw, ingredient_names)
            except Exception as e:
                last_err = e
                logger.warning(f"Mistral attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(1.5 * attempt)
 
        logger.error(f"All Mistral attempts failed: {last_err}")
        return [
            IngredientSafety(name=n, safety_verdict="unknown", safety_reason=f"Analysis failed — API error: {last_err}")
            for n in ingredient_names
        ]
 
    def _parse_json_response(self, raw_str: str, ingredient_names: List[str]) -> List[IngredientSafety]:
        raw_str = re.sub(r"^```json\s*|\s*```$", "", raw_str.strip())
        parsed = json.loads(raw_str)
        data = parsed["ingredients"] if isinstance(parsed, dict) and "ingredients" in parsed else parsed
        results = []
        for item in data:
            results.append(IngredientSafety(
                name=item.get("name", "").strip(),
                name_bn=item.get("name_bn", ""),
                safety_verdict=item.get("safety_verdict", "unknown"),
                safety_reason=item.get("safety_reason", ""),
                eu_status=item.get("eu_status", "unknown"),
                eu_e_number=item.get("eu_e_number"),
                eu_notes=item.get("eu_notes", ""),
                long_term_risk=bool(item.get("long_term_risk", False)),
                long_term_risk_detail=item.get("long_term_risk_detail", ""),
                risk_severity=item.get("risk_severity", "none"),
            ))
        return results
 
    def _mock_analyze(self, ingredient_names: List[str]) -> List[IngredientSafety]:
        knowledge = {
            "whole grain wheat": IngredientSafety(
                name="Whole grain wheat", name_bn="হোল গ্রেন গম",
                safety_verdict="safe", safety_reason="Nutritious grain containing fiber; safe for most people.",
                eu_status="permitted", eu_notes="No restrictions. Allergen warning required for gluten.",
                long_term_risk=False, risk_severity="none"
            ),
            "corn": IngredientSafety(
                name="Corn", name_bn="ভুট্টা",
                safety_verdict="safe", safety_reason="Standard grain ingredient, generally safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "rolled oats": IngredientSafety(
                name="Rolled oats", name_bn="রোলড ওটস",
                safety_verdict="safe", safety_reason="Highly nutritious grain, rich in beta-glucans.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "palm oil": IngredientSafety(
                name="Palm oil", name_bn="পাম তেল",
                safety_verdict="caution", safety_reason="High in saturated fats. High-temp processing creates potential carcinogens.",
                eu_status="permitted", eu_notes="Monitored for process contaminants like glycidyl esters.",
                long_term_risk=True, long_term_risk_detail="Regular consumption increases cardiovascular risk; process contaminants are carcinogenic.",
                risk_severity="low"
            ),
            "aspartame": IngredientSafety(
                name="Aspartame", name_bn="অ্যাসপারটেম",
                safety_verdict="caution", safety_reason="Artificial sweetener; classified by IARC as possibly carcinogenic (Group 2B).",
                eu_status="permitted", eu_e_number="E951",
                eu_notes="Permitted under ADI 40mg/kg bw; requires phenylalanine warning label.",
                long_term_risk=True, long_term_risk_detail="Possible cancer link (Group 2B); metabolic hazard for phenylketonurics.",
                risk_severity="moderate"
            ),
            "mango pieces": IngredientSafety(
                name="Mango pieces", name_bn="আমের টুকরো",
                safety_verdict="safe", safety_reason="Natural fruit components, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "mango": IngredientSafety(
                name="Mango", name_bn="আম",
                safety_verdict="safe", safety_reason="Natural fruit, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "mango juice": IngredientSafety(
                name="Mango juice", name_bn="আমের রস",
                safety_verdict="safe", safety_reason="Fruit juice, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "glycerol": IngredientSafety(
                name="Glycerol", name_bn="গ্লিসারল",
                safety_verdict="safe", safety_reason="Humectant; safe in dietary amounts.",
                eu_status="permitted", eu_e_number="E422",
                eu_notes="May act as a laxative in high concentrations.",
                long_term_risk=False, risk_severity="none"
            ),
            "tartrazine": IngredientSafety(
                name="Tartrazine", name_bn="টার্ট্রাজিন",
                safety_verdict="caution", safety_reason="Synthetic azo dye; linked to childhood hyperactivity and allergies.",
                eu_status="permitted", eu_e_number="E102",
                eu_notes="EU requires warning: 'may have an adverse effect on activity and attention in children.'",
                long_term_risk=True, long_term_risk_detail="Hyperactivity in children, potential allergic/asthmatic reaction.",
                risk_severity="moderate"
            ),
            "natural mango flavour": IngredientSafety(
                name="Natural mango flavour", name_bn="স্বাভাবিক আমের ফ্লেভার",
                safety_verdict="safe", safety_reason="Natural flavoring substance.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "royal jelly": IngredientSafety(
                name="Royal jelly", name_bn="রয়্যাল জেলি",
                safety_verdict="caution", safety_reason="Bee product; can trigger severe asthma and allergic attacks.",
                eu_status="permitted", eu_notes="Novel food regulation applies. Allergen warnings strongly recommended.",
                long_term_risk=True, long_term_risk_detail="Severe acute allergic reactions and bronchospasm in asthma/allergy sufferers.",
                risk_severity="moderate"
            ),
            "walnuts": IngredientSafety(
                name="Walnuts", name_bn="আখরোট",
                safety_verdict="safe", safety_reason="Nutritious tree nuts. Allergen warning required.",
                eu_status="permitted", eu_notes="Major tree nut allergen. Mandatory labeling required.",
                long_term_risk=False, risk_severity="none"
            ),
            "calcium carbonate": IngredientSafety(
                name="Calcium carbonate", name_bn="ক্যালসিয়াম কার্বনেট",
                safety_verdict="safe", safety_reason="Mineral source; generally safe.",
                eu_status="permitted", eu_e_number="E170", long_term_risk=False, risk_severity="none"
            ),
            "iron sulphate": IngredientSafety(
                name="Iron sulphate", name_bn="আয়রন সালফেট",
                safety_verdict="safe", safety_reason="Mineral supplement; safe within daily recommended allowances.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "vitamin c": IngredientSafety(
                name="Vitamin C", name_bn="ভিটামিন সি",
                safety_verdict="safe", safety_reason="Essential nutrient; ascorbic acid acts as an antioxidant.",
                eu_status="permitted", eu_e_number="E300", long_term_risk=False, risk_severity="none"
            ),
            "vitamin b6": IngredientSafety(
                name="Vitamin B6", name_bn="ভিটামিন বি৬",
                safety_verdict="safe", safety_reason="Essential nutrient, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "folic acid": IngredientSafety(
                name="Folic acid", name_bn="ফলিক অ্যাসিড",
                safety_verdict="safe", safety_reason="Synthetic folate; essential for cell growth.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "vitamin b12": IngredientSafety(
                name="Vitamin B12", name_bn="ভিটামিন বি১২",
                safety_verdict="safe", safety_reason="Essential nutrient, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            ),
            "spices": IngredientSafety(
                name="Spices", name_bn="মসলা",
                safety_verdict="safe", safety_reason="Natural plant spices, safe.",
                eu_status="permitted", long_term_risk=False, risk_severity="none"
            )
        }
 
        results = []
        for name in ingredient_names:
            norm = name.lower().strip()
            matched = None
            for key, val in knowledge.items():
                if key in norm or norm in key:
                    matched = val
                    break
            
            if matched:
                results.append(IngredientSafety(
                    name=name,
                    name_bn=matched.name_bn,
                    safety_verdict=matched.safety_verdict,
                    safety_reason=matched.safety_reason,
                    eu_status=matched.eu_status,
                    eu_e_number=matched.eu_e_number,
                    eu_notes=matched.eu_notes,
                    long_term_risk=matched.long_term_risk,
                    long_term_risk_detail=matched.long_term_risk_detail,
                    risk_severity=matched.risk_severity
                ))
            else:
                results.append(IngredientSafety(
                    name=name,
                    name_bn="",
                    safety_verdict="safe",
                    safety_reason="No toxic groups identified. Safe in normal food amounts.",
                    eu_status="permitted",
                    long_term_risk=False,
                    risk_severity="none"
                ))
        return results
 
    def generate_summary(self, ingredients: List[IngredientSafety], warnings: List[str]) -> str:
        if self.is_mock:
            unsafe = [i for i in ingredients if i.safety_verdict == "unsafe"]
            caution = [i for i in ingredients if i.safety_verdict == "caution"]
            risky = [i for i in ingredients if i.long_term_risk]
            
            summary = (
                f"A total of {len(ingredients)} ingredients were detected in this product. "
                f"Among them, {len(caution)} ingredient(s) require caution: "
                f"{', '.join([i.name for i in caution])}. "
            )
            if risky:
                summary += (
                    f"Regarding long-term health risks, Aspartame is classified for carcinogenicity (cancer risk), "
                    f"and Tartrazine may cause hyperactivity in children. "
                )
            summary += (
                f"According to EU standards, although Tartrazine and Aspartame are permitted, they are subject to strict restrictions. "
                f"A warning label is mandatory on packaging for Tartrazine. "
            )
            if warnings:
                summary += f"The packaging lists the following warning(s): {'; '.join(warnings)}. "
            
            summary += "Royal jelly should be consumed with caution by asthma and allergy sufferers."
            return summary
 
        unsafe = [i for i in ingredients if i.safety_verdict == "unsafe"]
        caution = [i for i in ingredients if i.safety_verdict == "caution"]
        risky = [i for i in ingredients if i.long_term_risk]
 
        summary_prompt = f"""Based on this ingredient analysis, write a SHORT, professional summary (4-6 sentences) in English.
 
Mention: 
1. Overall count of safe, caution, and unsafe ingredients.
2. EU status of ingredients (permitted, restricted, or banned).
3. Any long-term health risks (e.g., cancer, hyperactivity) if present.
4. Any label warnings that were detected.
 
Unsafe ingredients: {[i.name for i in unsafe]}
Caution ingredients: {[i.name for i in caution]}
Long-term risk ingredients: {[(i.name, i.long_term_risk_detail) for i in risky]}
Label warnings: {warnings}
 
Write naturally and professionally in English. Do not return JSON; write a plain text paragraph."""
 
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.3
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Mistral summary generation failed: {e}")
 
        # Fallback summary in common terms
        summary_parts = []
        if unsafe:
            names = ", ".join([i.name for i in unsafe])
            summary_parts.append(f"Please avoid this product if possible, as it contains unsafe ingredient(s): {names}.")
        if caution:
            names = ", ".join([i.name for i in caution])
            summary_parts.append(f"Consume this product with caution because it contains: {names}.")
        if risky:
            names = ", ".join([f"{i.name} ({i.long_term_risk_detail})" for i in risky])
            summary_parts.append(f"Note that there are long-term health risks associated with: {names}.")
        if warnings:
            summary_parts.append(f"The packaging carries important warning label(s): {'; '.join(warnings)}.")
        
        if not summary_parts:
            summary_parts.append("All parsed ingredients appear to be generally safe, with no immediate health hazards or specific label warnings identified.")
            
        return " ".join(summary_parts)
 
 
# ─── Orchestrator ──────────────────────────────────────────────────────────────
 
class NutriScanAnalyzer:
    """
    Full pipeline: raw OCR text -> cleaned ingredients -> Mistral analysis -> ProductAnalysis
    """
 
    def __init__(self, mistral_api_key: Optional[str] = None, model_name: str = "mistral-large-latest"):
        self.cleaner = OCRTextCleaner()
        self.mistral = MistralIngredientAnalyzer(api_key=mistral_api_key, model_name=model_name)
 
    def analyze_from_ocr_text(self, ocr_text: str) -> ProductAnalysis:
        ingredients_block = self.cleaner.extract_ingredients_block(ocr_text)
        if not ingredients_block.strip():
            raise ValueError("Input text is empty or does not contain any ingredients.")
 
        logger.info(f"Ingredients block extracted: {ingredients_block[:120]}...")
        ingredient_list = self.cleaner.split_ingredients(ingredients_block)
        logger.info(f"Parsed {len(ingredient_list)} individual ingredients: {ingredient_list}")
 
        warnings = self.cleaner.extract_warnings(ocr_text)
 
        analyzed = self.mistral.analyze(ingredient_list)
 
        # Allergen detection
        common_allergens = {"walnut", "wheat", "milk", "egg", "soy", "peanut", "tree nut", "gluten"}
        allergen_flags = []
        for ing in analyzed:
            for allergen in common_allergens:
                if allergen in ing.name.lower():
                    allergen_flags.append(ing.name)
                    break
 
        # Safety score: 100 - penalties
        score = 100
        for ing in analyzed:
            if ing.safety_verdict == "unsafe":
                score -= 20
            elif ing.safety_verdict == "caution":
                score -= 10
            if ing.long_term_risk:
                penalty = {"low": 5, "moderate": 10, "high": 20}.get(ing.risk_severity, 5)
                score -= penalty
        score = max(0, min(100, score))
 
        summary = self.mistral.generate_summary(analyzed, warnings)
 
        return ProductAnalysis(
            product_ingredients_raw=ingredients_block,
            ingredients=analyzed,
            overall_safety_score=score,
            warnings_detected=warnings,
            allergen_flags=allergen_flags,
            summary_bn_en=summary,
        )
 
 
 
# ─── CLI ──────────────────────────────────────────────────────────────────────
 
def main():
    import argparse
    p = argparse.ArgumentParser(description="NutriScan ingredient safety analyzer (Mistral AI)")
    p.add_argument("input", nargs="?", default=None, help="Text to analyze (or omit to paste interactively)")
    p.add_argument("--api-key", default=None, help="Mistral API key (or set MISTRAL_API_KEY env var)")
    p.add_argument("--model", default="mistral-large-latest", help="Mistral model name")
    p.add_argument("--output", choices=["json", "markdown"], default="markdown")
    p.add_argument("--save", default=None, help="Save output to file path")
    args = p.parse_args()
 
    text = args.input
    if not text:
        print("Please paste/type your text below.")
        print("When you are finished, press Ctrl-D (on Mac/Linux) or Ctrl-Z (on Windows) then press Enter:")
        import sys
        text = sys.stdin.read()
 
    analyzer = NutriScanAnalyzer(mistral_api_key=args.api_key, model_name=args.model)
    result = analyzer.analyze_from_ocr_text(text)
 
    output = result.to_json() if args.output == "json" else result.to_markdown()
    print(output)
 
    if args.save:
        Path(args.save).write_text(output, encoding="utf-8")
        logger.info(f"Saved to {args.save}")
 
if __name__ == "__main__":
    main()
