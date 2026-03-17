import json
from pathlib import Path

from datasets import load_dataset, DatasetDict
from tqdm import tqdm


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_split(s: str) -> str:
    s = str(s).strip().lower()
    if s in {"val", "valid", "validation", "dev"}:
        return "validation"
    return s


def get_dataset():
    obj = load_dataset("nlphuji/flickr30k")

    if isinstance(obj, DatasetDict):
        # 这个数据集常见情况是只有一个 HF split，真正的 train/val/test 在列里
        first_key = list(obj.keys())[0]
        ds = obj[first_key]
        print(f"Loaded DatasetDict with split: {first_key}")
        return ds

    return obj


def main():
    project_root = Path(__file__).resolve().parents[1]

    images_root = project_root / "data" / "raw" / "flickr30k_hf" / "flickr30k-images"
    out_json = project_root / "data" / "processed" / "flickr30k" / "flickr30k.json"

    ensure_dir(images_root)
    ensure_dir(out_json.parent)

    print("Loading dataset from Hugging Face...")
    ds = get_dataset()

    print(f"Loaded {len(ds)} image-level rows.")
    print("Exporting images + annotation json...")

    rows = []
    split_counter = {}
    text_id = 0

    for sample in tqdm(ds, total=len(ds)):
        filename = sample["filename"]
        img_id = int(sample["img_id"])
        split = normalize_split(sample["split"])

        split_counter[split] = split_counter.get(split, 0) + 1

        image = sample["image"]
        save_path = images_root / filename
        if not save_path.exists():
            image.save(save_path)

        captions = sample["caption"]
        for cap in captions:
            rows.append(
                {
                    "image": f"flickr30k-images/{filename}",
                    "caption": str(cap),
                    "image_id": img_id,
                    "text_id": text_id,
                    "split": split,
                }
            )
            text_id += 1

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved images to: {images_root}")
    print(f"Saved json to  : {out_json}")
    print(f"Total caption-level rows: {len(rows)}")
    print("Image-level split counts:", split_counter)

    train_n = sum(1 for x in rows if x["split"] == "train")
    val_n = sum(1 for x in rows if x["split"] == "validation")
    test_n = sum(1 for x in rows if x["split"] == "test")
    print("Caption-level split counts:")
    print("  train      =", train_n)
    print("  validation =", val_n)
    print("  test       =", test_n)


if __name__ == "__main__":
    main()