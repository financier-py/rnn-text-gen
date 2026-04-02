import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'war_and_peace_en.txt'
CLEAN_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'clean_text.txt'
VOCAB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'vocab.json'

def load_text(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text: str) -> str:
    text = text.lower()

    replacements = {'é': 'e', 'ê': 'e', 'à': 'a', 'ä': 'a'}
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    text = re.sub(r'[^a-z\n .,!?"\'\(\)\-:;]', ' ', text)
    text = re.sub(r'\b(chapter|book) [ivxlcdm]+\b', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text.strip()
    return text

def build_vocab(text: str) -> list[str]:
    return sorted(list(set(text)))

def save_artifacts(text: str, vocab: list[str]) -> None:
    with open(CLEAN_DATA_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    
    vocab_metadata = {
        'vocab_size': len(vocab),
        'chars': vocab
    }

    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_metadata, f, ensure_ascii=False, indent=4)

def main():
    raw_text = load_text(RAW_DATA_PATH)
    
    cleaned_text = clean_text(raw_text)

    vocab = build_vocab(cleaned_text)

    save_artifacts(cleaned_text, vocab)

if __name__ == '__main__':
    main()