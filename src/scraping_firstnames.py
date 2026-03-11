"""
Scrape first-name information from Behind the Name.

This script:
- reads a small list of first names
- builds the corresponding Behind the Name URL
- downloads the page
- extracts:
    * first name
    * page title
    * meaning/history text
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
OUTPUT_FILE = RESULTS_DIR / "firstnames_dataset.json"

BASE_URL = "https://www.behindthename.com/name/"


def save_json(data: Any, file_path: Path) -> None:
    """Save data as formatted JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


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


def clean_text(text: str) -> str:
    """Normalize extracted text."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_meaning_history(soup: BeautifulSoup) -> str:
    """
    Extract the 'Meaning & History' section as best as possible.

    The structure of pages can vary, so this function uses a few fallback rules.
    """
    full_text = soup.get_text("\n", strip=True)

    start_markers = [
        "Meaning & History",
        "Meaning & History Expand Links",
    ]

    for marker in start_markers:
        if marker in full_text:
            section = full_text.split(marker, 1)[1]

            end_markers = [
                "User Submission",
                "Related Names",
                "Popularity",
                "Comments",
                "Namesakes",
                "Name Days",
                "Entry updated",
            ]

            end_index = len(section)
            for end_marker in end_markers:
                pos = section.find(end_marker)
                if pos != -1 and pos < end_index:
                    end_index = pos

            section = section[:end_index]
            return clean_text(section)

    return ""



def parse_name_page(html: str, first_name: str, url: str) -> Dict[str, Any]:
    """Parse a Behind the Name page and extract structured fields."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    meaning_history = extract_meaning_history(soup)

    origin = ""
    meaning = ""
    description = meaning_history

    if meaning_history:
        sentences = re.split(r"(?<=[.!?])\s+", meaning_history)

        if sentences:
            first_sentence = sentences[0].strip()
            meaning = first_sentence

            origin_patterns = [
                r"Hebrew",
                r"Latin",
                r"Greek",
                r"French",
                r"German",
                r"Arabic",
                r"Czech",
                r"English",
            ]

            for pattern in origin_patterns:
                if re.search(pattern, first_sentence, re.IGNORECASE):
                    origin = pattern
                    break

        if len(sentences) > 1:
            description = " ".join(sentences[1:]).strip()

    return {
        "first_name": first_name,
        "url": url,
        "source": "Behind the Name",
        "title": clean_text(title),
        "origin": origin,
        "meaning": clean_text(meaning),
        "description": clean_text(description),
    }
def scrape_first_names(first_names: List[str]) -> List[Dict[str, Any]]:
    """Scrape several first-name pages."""
    results: List[Dict[str, Any]] = []

    for first_name in first_names:
        slug = first_name.strip().lower()
        url = f"{BASE_URL}{slug}"

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
                    "error": str(error),
                }
            )
            print(f"ERROR: {first_name} -> {error}")

    return results


def main() -> None:
    """Run the scraping pipeline."""
    first_names = ["Adam", "Sarah", "Marie", "Lucas"]

    print("Starting first-name scraping...")
    results = scrape_first_names(first_names)

    print("Saving results...")
    save_json(results, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()