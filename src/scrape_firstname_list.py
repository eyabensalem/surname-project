"""
Scrape a list of first names from OrigineNom.

Step 1 of the first-name extension:
- visit the "Liste des prénoms" pages
- extract first-name labels and profile URLs
- save them to JSON

Later, this list will be used to scrape each first-name page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_LIST_URL = "https://originenom.com/liste-des-prenoms/"

from config import FIRSTNAMES_LIST_FILE

OUTPUT_FILE = FIRSTNAMES_LIST_FILE

def save_json(data: Any, file_path: Path) -> None:
    """Save data as formatted JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    """Normalize text."""
    text = re.sub(r"\s+", " ", text).strip()
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


def extract_firstname_links(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    Extract first-name links from one list page.

    We keep only URLs that look like:
    /origine-du-prenom/<slug>/
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        label = clean_text(link.get_text(" ", strip=True))

        if not label:
            continue
        # Filter very noisy entries
        if len(label) < 2:
            continue

        if any(char.isdigit() for char in label):
            continue

        # ignore extremely long entries
        if len(label) > 30:
            continue

        # ignore entries with too many words
        if len(label.split()) > 3:
            continue
        
        if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ'\- ]+$", label):
            continue
                    # ignore entries that are only separators or too generic
        if label.lower() in {"origine", "signification", "prenom", "prénoms"}:
            continue
        full_url = urljoin(base_url, href)

        if "/origine-du-prenom/" not in full_url:
            continue

        # Ignore generic search page
        if full_url.rstrip("/").endswith("/origine-du-prenom"):
            continue

        # Keep clean profile-like URLs only
        if not re.search(r"/origine-du-prenom/[^/]+/?$", full_url):
            continue

        key = f"{label.lower()}|{full_url.lower()}"
        if key in seen:
            continue

        seen.add(key)
        results.append(
            {
                "first_name": label,
                "url": full_url,
            }
        )

    return results


def build_paginated_urls(num_pages: int) -> List[str]:
    """
    Build paginated list URLs.

    Page 1 uses the base URL.
    Next pages usually follow the pattern /page/<n>/.
    """
    urls = [BASE_LIST_URL]

    for page_num in range(2, num_pages + 1):
        urls.append(urljoin(BASE_LIST_URL, f"page/{page_num}/"))

    return urls


def scrape_firstname_list(num_pages: int = 3) -> List[Dict[str, str]]:
    """Scrape several pages of the firstname list."""
    all_results: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()

    urls = build_paginated_urls(num_pages)

    for url in urls:
        try:
            print(f"Fetching: {url}")
            html = fetch_page(url)
            page_results = extract_firstname_links(html, url)

            for item in page_results:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_results.append(item)

            print(f"  -> {len(page_results)} names found")
        except Exception as error:
            print(f"ERROR on {url}: {error}")

    return all_results


def main() -> None:
    """Run the firstname list scraper."""
    print("Starting firstname list scraping from OrigineNom...")
    firstnames = scrape_firstname_list(num_pages=3)

    print(f"Total unique first names collected: {len(firstnames)}")
    save_json(firstnames, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()