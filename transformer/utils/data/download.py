"""
IWSLT 2016 Dataset Download.

Downloads and extracts the IWSLT 2016 German-English dataset.
"""

import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple


IWSLT_URL = "https://wit3.fbk.eu/archive/2016-01/texts/de/en/de-en.tgz"


def download_iwslt2016(data_dir: str = "./data") -> Path:
    """
    Download and extract IWSLT 2016 De-En dataset.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to the extracted dataset directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / "de-en.tgz"
    extract_dir = data_dir / "de-en"

    # Check if already extracted
    if extract_dir.exists():
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir

    # Download archive
    if not archive_path.exists():
        print(f"Downloading IWSLT 2016 De-En dataset...")
        print(f"URL: {IWSLT_URL}")

        try:
            urllib.request.urlretrieve(IWSLT_URL, archive_path, _download_progress)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nFailed to download from {IWSLT_URL}")
            print(f"Error: {e}")
            print("\nPlease download manually and place at:", archive_path)
            raise

    # Extract archive
    print(f"Extracting to {extract_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print("Extraction complete!")

    return extract_dir


def _download_progress(block_num: int, block_size: int, total_size: int):
    """Progress callback for urllib download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")


def parse_xml_file(xml_path: Path) -> List[str]:
    """
    Parse IWSLT XML file and extract sentences.

    Args:
        xml_path: Path to the XML file

    Returns:
        List of sentences
    """
    sentences = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all <seg> elements
        for seg in root.iter('seg'):
            if seg.text:
                text = seg.text.strip()
                if text:
                    sentences.append(text)
    except ET.ParseError:
        # Handle malformed XML by reading as text
        with open(xml_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('<seg'):
                    # Extract text between <seg> tags
                    start = line.find('>') + 1
                    end = line.rfind('<')
                    if start < end:
                        text = line[start:end].strip()
                        if text:
                            sentences.append(text)

    return sentences


def parse_txt_file(txt_path: Path) -> List[str]:
    """
    Parse plain text file with one sentence per line.

    Args:
        txt_path: Path to the text file

    Returns:
        List of sentences
    """
    sentences = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                sentences.append(line)
    return sentences


def load_iwslt_split(
    data_dir: Path,
    split: str = "train"
) -> Tuple[List[str], List[str]]:
    """
    Load a specific split of IWSLT dataset.

    Args:
        data_dir: Path to the extracted de-en directory
        split: One of "train", "valid", "test"

    Returns:
        Tuple of (source_sentences, target_sentences)
    """
    data_dir = Path(data_dir)

    if split == "train":
        # Training data is in multiple TED talk files
        de_sentences = []
        en_sentences = []

        # Find all training files
        train_dir = data_dir / "de-en"
        if not train_dir.exists():
            train_dir = data_dir

        # Look for train.tags.de-en.* files
        for de_file in sorted(train_dir.glob("train.tags.de-en.de")):
            en_file = de_file.with_suffix('.en')
            if en_file.exists():
                de_sents = parse_txt_file(de_file)
                en_sents = parse_txt_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)

        # Also check for XML format
        for de_file in sorted(train_dir.glob("*.de.xml")):
            en_file = de_file.with_name(de_file.name.replace('.de.xml', '.en.xml'))
            if en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)

        return de_sentences, en_sentences

    elif split == "valid":
        # Validation data
        valid_patterns = [
            ("IWSLT16.TED.tst2013.de-en.de.xml", "IWSLT16.TED.tst2013.de-en.en.xml"),
            ("IWSLT16.TED.dev2010.de-en.de.xml", "IWSLT16.TED.dev2010.de-en.en.xml"),
        ]

        de_sentences = []
        en_sentences = []

        for de_pattern, en_pattern in valid_patterns:
            de_file = data_dir / "de-en" / de_pattern
            en_file = data_dir / "de-en" / en_pattern

            if not de_file.exists():
                de_file = data_dir / de_pattern
                en_file = data_dir / en_pattern

            if de_file.exists() and en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)
                break

        return de_sentences, en_sentences

    elif split == "test":
        # Test data
        test_patterns = [
            ("IWSLT16.TED.tst2014.de-en.de.xml", "IWSLT16.TED.tst2014.de-en.en.xml"),
            ("IWSLT16.TED.tst2015.de-en.de.xml", "IWSLT16.TED.tst2015.de-en.en.xml"),
        ]

        de_sentences = []
        en_sentences = []

        for de_pattern, en_pattern in test_patterns:
            de_file = data_dir / "de-en" / de_pattern
            en_file = data_dir / "de-en" / en_pattern

            if not de_file.exists():
                de_file = data_dir / de_pattern
                en_file = data_dir / en_pattern

            if de_file.exists() and en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)
                break

        return de_sentences, en_sentences

    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'valid', or 'test'.")
