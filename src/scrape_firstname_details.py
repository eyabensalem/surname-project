"""
Scrape first-name detail pages from OrigineNom.

Step 2 of the first-name extension:
- load firstnames_list.json
- visit each first-name page
- extract structured fields
- save results to firstnames_dataset.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup




from config import FIRSTNAMES_LIST_FILE, FIRSTNAMES_DATASET_FILE

INPUT_FILE = FIRSTNAMES_LIST_FILE
OUTPUT_FILE = FIRSTNAMES_DATASET_FILE
def load_json(file_path: Path) -> Any:
    """Load JSON content."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: Path) -> None:
    """Save JSON content."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    """Normalize extracted text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    return text


def fetch_page(url: str, timeout: int = 15) -> str:
    """Download page HTML."""
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    return response.text


def extract_text_blocks(soup: BeautifulSoup) -> List[str]:
    """Extract visible text blocks."""
    blocks: List[str] = []

    for tag in soup.find_all(["p", "li", "h2", "h3", "strong"]):
        text = tag.get_text(" ", strip=True)
        text = clean_text(text)
        if text:
            blocks.append(text)

    return blocks


def extract_field_after_label(blocks: List[str], label: str) -> str:
    """
    Extract content after labels such as:
    'Origine : ...'
    'Signification : ...'
    """
    pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE)

    for block in blocks:
        match = pattern.match(block)
        if match:
            return clean_text(match.group(1))

    return ""


def extract_description(blocks: List[str], first_name: str) -> str:
    """
    Extract a fallback description:
    - prefer a paragraph mentioning the first name
    - otherwise keep the first substantial paragraph
    """
    first_name_lower = first_name.lower()
    candidates: List[str] = []

    for block in blocks:
        lower = block.lower()

        if len(block) >= 80:
            candidates.append(block)

        if first_name_lower in lower and len(block) >= 50:
            return block

    return candidates[0] if candidates else ""
def compute_quality_score(origin: str, meaning: str, description: str) -> int:
    """
    Compute a simple quality score for a firstname record.
    """
    score = 0

    if origin.strip():
        score += 1

    if meaning.strip():
        score += 1

    if description.strip() and len(description.strip()) >= 60:
        score += 1

    return score
def infer_origin_from_text(text: str) -> str:
    """
    Try to infer a normalized origin from free text.
    """
    lowered = text.lower()

    origin_patterns = {
        "Hébraïque": [r"origine[s]?\s+hébra", r"racine[s]?\s+hébra", r"hébreu", r"hébraïque"],
        "Arabe": [r"origine[s]?\s+arabe", r"racine[s]?\s+arabe", r"\barabe\b"],
        "Latin": [r"origine[s]?\s+latine?", r"racine[s]?\s+latine?", r"\blatin\b"],
        "Grecque": [r"origine[s]?\s+grec", r"racine[s]?\s+grec", r"\bgrec\b", r"grecque"],
        "Biblique": [r"\bbible\b", r"ancien testament", r"biblique"],
        "Française": [r"origine[s]?\s+fran", r"\bfrançais\b", r"\bfrançaise\b"],
    }

    for normalized_origin, patterns in origin_patterns.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                return normalized_origin

    return ""
def infer_meaning_from_text(text: str) -> str:
    """
    Try to infer a meaning from free text using common patterns.
    """
    patterns = [
        r"signifie\s+\"([^\"]+)\"",
        r"signifie\s+«([^»]+)»",
        r"signifie\s+“([^”]+)”",
        r"signifie\s+'([^']+)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    return ""
def parse_name_page(html: str, first_name: str, url: str) -> Dict[str, Any]:
    """Parse one first-name detail page."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    blocks = extract_text_blocks(soup)

    origin = extract_field_after_label(blocks, "Origine")
    meaning = extract_field_after_label(blocks, "Signification")
    description = extract_description(blocks, first_name)
    # Fallback extraction if structured fields are missing or badly extracted
    if not origin or len(origin.split()) > 8:
        inferred_origin = infer_origin_from_text(description)
        if inferred_origin:
            origin = inferred_origin

    if not meaning:
        inferred_meaning = infer_meaning_from_text(description)
        if inferred_meaning:
            meaning = inferred_meaning
    quality_score = compute_quality_score(origin, meaning, description)
    return {
        "first_name": first_name,
        "url": url,
        "source": "OrigineNom",
        "title": clean_text(title),
        "origin": origin,
        "meaning": meaning,
        "description": clean_text(description),
        "quality_score": quality_score,
    }
def is_valid_firstname_record(record: Dict[str, Any]) -> bool:
    """
    Keep only records with enough useful information.
    Reject generic search-like pages or surname-like entries.
    """
    if record.get("error"):
        return False

    first_name = record.get("first_name", "").strip()
    title = record.get("title", "").strip().lower()
    origin = record.get("origin", "").strip()
    meaning = record.get("meaning", "").strip()
    description = record.get("description", "").strip().lower()

    if len(first_name) < 2:
        return False

    # Reject generic search pages with no useful structured info
    if "recherche - origine nom" in title and not origin and not meaning:
        return False

    # Reject entries that explicitly say this is a surname
    if "nom de famille" in description:
        return False

    # Reject very weak records
    content_score = sum(bool(x) for x in [origin, meaning, description])

    if content_score == 0:
        return False

    if len(description) < 60 and not origin and not meaning:
        return False

    return True

def scrape_firstname_details(firstnames: List[Dict[str, str]], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape detail pages for the first N first names.
    """
    results: List[Dict[str, Any]] = []

    for item in firstnames[:limit]:
        first_name = item["first_name"]
        url = item["url"]

        try:
            html = fetch_page(url)
            parsed = parse_name_page(html, first_name, url)

            if is_valid_firstname_record(parsed):
                results.append(parsed)
                print(f"OK: {first_name}")
            else:
                print(f"SKIPPED: {first_name} (insufficient content)")
        except Exception as error:
            results.append(
                {
                    "first_name": first_name,
                    "url": url,
                    "source": "OrigineNom",
                    "error": str(error),
                }
            )
            print(f"ERROR: {first_name} -> {error}")

    return results


def main() -> None:
    """Run firstname detail scraping."""
    print("Loading firstname list...")
    firstnames = load_json(INPUT_FILE)

    print("Scraping firstname details...")
    results = scrape_firstname_details(firstnames, limit=10)

    print("Saving results...")
    save_json(results, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()