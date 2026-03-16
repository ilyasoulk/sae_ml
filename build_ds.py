import json
import os
from datasets import load_dataset

def build_large_flores():
    # The exact union of Paper Languages + Aya Languages
    flores_mapping = {
        "en": "eng",  # Base
        
        # Paper Languages
        "es": "spa",  # Spanish
        "fr": "fra",  # French
        "ja": "jpn",  # Japanese
        "ko": "kor",  # Korean
        "pt": "por",  # Portuguese
        "th": "tha",  # Thai
        "zh": "cmn",  # Mandarin Chinese
        "vi": "vie",  # Vietnamese 
        "ar": "arb",  # Arabic 
        
        # Additional Custom Aya Languages
        "yor": "yor", # Yoruba
        "tam": "tam", # Tamil
        "pan": "pan", # Panjabi
        "sin": "sin", # Sinhala
        "som": "som", # Somali
        "tel": "tel", # Telugu
        "guj": "guj", # Gujarati
        "zsm": "zsm"  # Standard Malay
    }

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "parallel_corpus_large.jsonl")

    print("Loading FLORES+ from Hugging Face...")
    try:
        dataset = load_dataset("openlanguagedata/flores_plus", split="devtest")
    except Exception as e:
        print("\nERROR: Could not load the dataset.")
        raise e
    
    lang_texts = {}
    target_length = 0
    
    for lang_short, iso_code in flores_mapping.items():
        print(f"Extracting {lang_short.upper()} ({iso_code})...")
        lang_data = dataset.filter(lambda x: x["iso_639_3"] == iso_code)
        lang_texts[lang_short] = [row["text"] for row in lang_data]
        
        if lang_short == "en":
            target_length = len(lang_texts[lang_short])
            print(f"  -> Extracted {target_length} sentences (Base Length).")
        else:
            # Handle languages with multiple scripts in FLORES (like Arabic or Chinese)
            if len(lang_texts[lang_short]) > target_length:
                print(f"  -> Found {len(lang_texts[lang_short])} sentences. Truncating to {target_length} (primary script).")
                lang_texts[lang_short] = lang_texts[lang_short][:target_length]
            else:
                print(f"  -> Extracted {len(lang_texts[lang_short])} sentences.")

    # Final safety check
    for lang, texts in lang_texts.items():
        assert len(texts) == target_length, f"Alignment error: {lang} has {len(texts)} sentences, expected {target_length}."

    print(f"\nAligning translations and writing to {out_file}...")
    
    aligned_data = []
    for i in range(target_length):
        record = {}
        for lang in flores_mapping.keys():
            record[lang] = lang_texts[lang][i]
        aligned_data.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in aligned_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Success! Parallel corpus is ready. Total aligned sentence groups: {len(aligned_data)}")

if __name__ == "__main__":
    build_large_flores()