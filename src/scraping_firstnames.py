"""
Scrape first-name information from OrigineNom.

This script:
- reads a list of first names
- builds OrigineNom URLs
- downloads each page
- extracts origin / meaning / description
- saves results to JSON
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


RESULTS_DIR = Path("results")
BASE_URL = "https://originenom.com/origine-du-prenom/"

from config import FIRSTNAMES_DATASET_FILE

OUTPUT_FILE = FIRSTNAMES_DATASET_FILE

def save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    return text


def fetch_page(url: str, timeout: int = 15) -> str:
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
    blocks: List[str] = []

    for tag in soup.find_all(["p", "li", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        text = clean_text(text)
        if text:
            blocks.append(text)

    return blocks


def extract_field_after_label(blocks: List[str], label: str) -> str:
    """
    Try to extract content after labels like:
    'Origine : ...' or 'Signification : ...'
    """
    pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE)

    for block in blocks:
        match = pattern.match(block)
        if match:
            return clean_text(match.group(1))

    return ""


def extract_description(blocks: List[str], first_name: str) -> str:
    """
    Fallback description: keep the first informative paragraph mentioning the name
    or the first substantial paragraph.
    """
    candidates = []

    for block in blocks:
        lower = block.lower()
        if len(block) > 80:
            candidates.append(block)
        if first_name.lower() in lower and len(block) > 50:
            return block

    return candidates[0] if candidates else ""


def parse_name_page(html: str, first_name: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    blocks = extract_text_blocks(soup)

    origin = extract_field_after_label(blocks, "Origine")
    meaning = extract_field_after_label(blocks, "Signification")
    description = extract_description(blocks, first_name)

    return {
        "first_name": first_name,
        "url": url,
        "source": "OrigineNom",
        "title": clean_text(title),
        "origin": origin,
        "meaning": meaning,
        "description": clean_text(description),
    }


def scrape_first_names(first_names: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for first_name in first_names:
        slug = first_name.strip().lower()
        url = f"{BASE_URL}{slug}/"

        try:
            html = fetch_page(url)
            parsed = parse_name_page(html, first_name, url)
            results.append(parsed)
            print(f"OK: {first_name}")
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
    first_names = ["Hedjem", "Fonte", "Nemalie", "Daic"]

    print("Starting first-name scraping from OrigineNom...")
    results = scrape_first_names(first_names)

    print("Saving results...")
    save_json(results, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()