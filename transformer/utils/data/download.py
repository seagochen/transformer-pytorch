"""
IWSLT 2016 Dataset Download.

Downloads and extracts the IWSLT 2016 German-English dataset.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple


KAGGLE_DATASET = "tttzof351/iwslt-2016-de-en"


def download_iwslt2016(data_dir: str = "./data") -> Path:
    """
    Download and extract IWSLT 2016 De-En dataset from Kaggle.

    Requires kaggle CLI to be installed and configured with API credentials.
    See: https://www.kaggle.com/docs/api

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to the extracted dataset directory
    """
    import subprocess
    import zipfile

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = data_dir / "de-en"

    # Check if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir

    archive_path = data_dir / "iwslt-2016-de-en.zip"

    # Download from Kaggle if archive doesn't exist
    if not archive_path.exists():
        print(f"Downloading IWSLT 2016 De-En dataset from Kaggle...")
        print(f"Dataset: {KAGGLE_DATASET}")

        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("Download complete!")
        except FileNotFoundError:
            print("\nError: kaggle CLI not found.")
            print("Please install it with: pip install kaggle")
            print("And configure your API credentials: https://www.kaggle.com/docs/api")
            raise
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to download from Kaggle: {e.stderr}")
            print("\nPlease ensure:")
            print("1. kaggle CLI is installed: pip install kaggle")
            print("2. API credentials are configured: ~/.kaggle/kaggle.json")
            print("3. Or download manually from: https://www.kaggle.com/datasets/tttzof351/iwslt-2016-de-en")
            print(f"   and place the zip file at: {archive_path}")
            raise

    # Extract archive
    print(f"Extracting to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction complete!")
    return extract_dir


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


def _find_data_dir(base_dir: Path) -> Path:
    """Find the actual data directory, handling nested structures."""
    # Check for nested 'data' directory (Kaggle format)
    nested_data = base_dir / "data"
    if nested_data.exists() and nested_data.is_dir():
        return nested_data

    # Check for nested 'de-en' directory
    nested_deen = base_dir / "de-en"
    if nested_deen.exists() and nested_deen.is_dir():
        # Check if there's a further nested 'data' directory
        nested_data = nested_deen / "data"
        if nested_data.exists() and nested_data.is_dir():
            return nested_data
        return nested_deen

    return base_dir


def load_iwslt_split(
    data_dir: Path,
    split: str = "train"
) -> Tuple[List[str], List[str]]:
    """
    Load a specific split of IWSLT dataset.

    For Kaggle dataset (which only contains training data), validation and test
    splits are created by reserving portions of the training data.

    Args:
        data_dir: Path to the extracted de-en directory
        split: One of "train", "valid", "test"

    Returns:
        Tuple of (source_sentences, target_sentences)
    """
    data_dir = Path(data_dir)
    actual_data_dir = _find_data_dir(data_dir)

    # Try to load from train.tags.de-en.* files (Kaggle format)
    de_file = actual_data_dir / "train.tags.de-en.de"
    en_file = actual_data_dir / "train.tags.de-en.en"

    if de_file.exists() and en_file.exists():
        de_sentences = parse_txt_file(de_file)
        en_sentences = parse_txt_file(en_file)

        if len(de_sentences) != len(en_sentences):
            # Truncate to matching length
            min_len = min(len(de_sentences), len(en_sentences))
            de_sentences = de_sentences[:min_len]
            en_sentences = en_sentences[:min_len]

        # Split data: 90% train, 5% valid, 5% test
        total = len(de_sentences)
        train_end = int(total * 0.90)
        valid_end = int(total * 0.95)

        if split == "train":
            return de_sentences[:train_end], en_sentences[:train_end]
        elif split == "valid":
            return de_sentences[train_end:valid_end], en_sentences[train_end:valid_end]
        elif split == "test":
            return de_sentences[valid_end:], en_sentences[valid_end:]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'valid', or 'test'.")

    # Fallback to original IWSLT format with separate files
    if split == "train":
        de_sentences = []
        en_sentences = []

        for de_file in sorted(actual_data_dir.glob("train.tags.de-en.de")):
            en_file = de_file.with_suffix('.en')
            if en_file.exists():
                de_sents = parse_txt_file(de_file)
                en_sents = parse_txt_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)

        for de_file in sorted(actual_data_dir.glob("*.de.xml")):
            en_file = de_file.with_name(de_file.name.replace('.de.xml', '.en.xml'))
            if en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    de_sentences.extend(de_sents)
                    en_sentences.extend(en_sents)

        return de_sentences, en_sentences

    elif split == "valid":
        valid_patterns = [
            ("IWSLT16.TED.tst2013.de-en.de.xml", "IWSLT16.TED.tst2013.de-en.en.xml"),
            ("IWSLT16.TED.dev2010.de-en.de.xml", "IWSLT16.TED.dev2010.de-en.en.xml"),
        ]

        for de_pattern, en_pattern in valid_patterns:
            de_file = actual_data_dir / de_pattern
            en_file = actual_data_dir / en_pattern

            if de_file.exists() and en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    return de_sents, en_sents

        return [], []

    elif split == "test":
        test_patterns = [
            ("IWSLT16.TED.tst2014.de-en.de.xml", "IWSLT16.TED.tst2014.de-en.en.xml"),
            ("IWSLT16.TED.tst2015.de-en.de.xml", "IWSLT16.TED.tst2015.de-en.en.xml"),
        ]

        for de_pattern, en_pattern in test_patterns:
            de_file = actual_data_dir / de_pattern
            en_file = actual_data_dir / en_pattern

            if de_file.exists() and en_file.exists():
                de_sents = parse_xml_file(de_file)
                en_sents = parse_xml_file(en_file)
                if len(de_sents) == len(en_sents):
                    return de_sents, en_sents

        return [], []

    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'valid', or 'test'.")
