import os, yaml, csv

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def convert_annotations(csv_in, csv_out, dim):
    with open(csv_in, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            filename = r["filename"]
            x = float(r["x"])
            y = float(r["y"])
            w = float(r["width"])
            h = float(r["height"])
            rows.append({
                "filename": filename,
                "x_min": x/dim,
                "y_min": y/dim,
                "x_max": (x+w)/dim,
                "y_max": (y+h)/dim
            })
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x_min", "y_min", "x_max", "y_max"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_out}")

def main():
    config = load_config()
    convert_annotations(config["annotations_csv"], config["annotations_coco_csv"], config.get("dimension", 256))

if __name__ == "__main__":
    main()
