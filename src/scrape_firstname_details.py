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


RESULTS_DIR = Path("results")
INPUT_FILE = RESULTS_DIR / "firstnames_list.json"
OUTPUT_FILE = RESULTS_DIR / "firstnames_dataset.json"


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


def parse_name_page(html: str, first_name: str, url: str) -> Dict[str, Any]:
    """Parse one first-name detail page."""
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