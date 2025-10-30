import os, time, json, math, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from pdf2image import convert_from_path

# ========== CONFIG ==========
from pathlib import Path

RAW_DIR   = Path("data/raw")
IMG_DIR   = Path("data/img")
OUT_DIR   = Path("outputs")
YOLO_WEIGHTS = Path("weights/table_detector.pt")  # optional
MAX_PAGES = 5   # limit per PDF
PDF_DPI   = 300
DET_THRESH = 0.50

FILES = [
    RAW_DIR/"File1.pdf",
    RAW_DIR/"File2.pdf",
    RAW_DIR/"File3.pdf",
    RAW_DIR/"Picture1.jpg",
    RAW_DIR/"Picture2.jpg",
    RAW_DIR/"Picture3.jpg",
]
# ===========================

# ---------- utils ----------
def ensure_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"tatr").mkdir(exist_ok=True, parents=True)
    (OUT_DIR/"ppstructure").mkdir(exist_ok=True, parents=True)
    (OUT_DIR/"yolo").mkdir(exist_ok=True, parents=True)

def pdf_to_images(pdf_path: Path) -> List[Path]:
    from pdf2image import convert_from_path
    pages = convert_from_path(str(pdf_path), dpi=PDF_DPI)
    out = []
    for i, p in enumerate(pages[:MAX_PAGES], 1):
        out_path = IMG_DIR/f"{pdf_path.stem}_p{i:03}.png"
        p.save(out_path)
        out.append(out_path)
    return out

def load_as_images(path: Path) -> List[Path]:
    if path.suffix.lower() == ".pdf":
        return pdf_to_images(path)
    else:
        # normalize to PNG copy (for consistent downstream)
        im = Image.open(path).convert("RGB")
        out = IMG_DIR/f"{path.stem}.png"
        im.save(out)
        return [out]

def draw_boxes(img_path: Path, boxes: List[Tuple[float,float,float,float]], color=(255,0,0), width=3) -> Path:
    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for (x1,y1,x2,y2) in boxes:
        dr.rectangle([x1,y1,x2,y2], outline=color, width=width)
    out_path = img_path.parent / f"{img_path.stem}_overlay.png"
    im.save(out_path)
    return out_path

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Model A: TATR ----------
def run_tatr(image_path: Path) -> Dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from transformers import AutoProcessor, AutoModelForSeq2SeqLM
    im = Image.open(image_path).convert("RGB")

    det_proc = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    det_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    t0 = time.time()
    inputs = det_proc(images=im, return_tensors="pt")
    det = det_model(**inputs)
    post = det_proc.post_process_object_detection(det, threshold=DET_THRESH,
                                                  target_sizes=[im.size[::-1]])[0]
    boxes = post["boxes"].detach().numpy().tolist()
    scores = post["scores"].detach().numpy().tolist()

    # structure model
    str_proc = AutoProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    str_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/table-transformer-structure-recognition")

    tables = []
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = map(int, b)
        crop = im.crop((x1,y1,x2,y2))
        s_inputs = str_proc(images=crop, text="predict table structure", return_tensors="pt")
        out_ids = str_model.generate(**s_inputs, max_new_tokens=1024)
        html = str_proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        tables.append({"bbox":[x1,y1,x2,y2], "score":float(scores[i]), "structure_html": html})

    elapsed = round(time.time()-t0,2)
    # overlay
    overlay = draw_boxes(image_path, [tuple(map(float,b)) for b in boxes], color=(255,0,0))
    return {"model":"tatr", "image": str(image_path), "time_sec": elapsed,
            "n_tables": len(tables), "tables": tables, "overlay": str(overlay)}

# ---------- Model B: PP-Structure ----------
_pp_cached = None
def _get_pp_engine():
    global _pp_cached
    if _pp_cached is None:
        from paddleocr import PPStructure
        _pp_cached = PPStructure(layout=True, show_log=False)  # uses table + layout + ocr
    return _pp_cached

def run_ppstructure(image_path: Path) -> Dict[str, Any]:
    import cv2
    img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    engine = _get_pp_engine()
    t0 = time.time()
    res = engine(img)
    elapsed = round(time.time()-t0,2)

    tables = []
    boxes = []
    for block in res:
        if block.get("type") == "table":
            info = block.get("res", {})
            html = info.get("html", None)
            bbox = block.get("bbox", None)  # may be None depending on version
            if bbox is None and "cell_bbox" in info:
                # fallback bbox from cells
                xs = [b[0] for b in info["cell_bbox"]] + [b[2] for b in info["cell_bbox"]]
                ys = [b[1] for b in info["cell_bbox"]] + [b[3] for b in info["cell_bbox"]]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            if bbox:
                boxes.append(tuple(map(float, bbox)))
            tables.append({"bbox": bbox, "html": html})
    overlay = draw_boxes(image_path, boxes, color=(0,128,255))
    return {"model":"ppstructure", "image": str(image_path), "time_sec": elapsed,
            "n_tables": len(tables), "tables": tables, "overlay": str(overlay)}

# ---------- Model C: YOLOv8 (optional) ----------
_yolo_cached = None
def _get_yolo():
    global _yolo_cached
    if _yolo_cached is not None:
        return _yolo_cached
    if not YOLO_WEIGHTS.exists():
        return None
    from ultralytics import YOLO
    _yolo_cached = YOLO(str(YOLO_WEIGHTS))
    return _yolo_cached

def run_yolo(image_path: Path) -> Dict[str, Any]:
    model = _get_yolo()
    if model is None:
        return {"model":"yolo", "image":str(image_path), "skipped":"no_weights"}
    im = Image.open(image_path).convert("RGB")
    t0 = time.time()
    res = model.predict(source=np.array(im), verbose=False)[0]
    elapsed = round(time.time()-t0,2)
    boxes=[]
    for b in res.boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        boxes.append((float(x1),float(y1),float(x2),float(y2)))
    overlay = draw_boxes(image_path, boxes, color=(0,255,0))
    return {"model":"yolo", "image":str(image_path), "time_sec": elapsed,
            "n_tables": len(boxes), "tables":[{"bbox":list(b)} for b in boxes], "overlay": str(overlay)}

# ---------- main ----------
def main():
    ensure_dirs()
    # expand to images
    page_images: List[Path] = []
    for f in FILES:
        if not f.exists():
            print(f"[WARN] Missing: {f}")
            continue
        imgs = load_as_images(f)
        page_images.extend(imgs)

    summary_rows = []
    # process each image with three models
    for img in page_images:
        print(f"\n=== Processing {img.name} ===")
        # TATR
        try:
            r1 = run_tatr(img)
            save_json(r1, OUT_DIR/"tatr"/f"{img.stem}.json")
            summary_rows.append({"image":img.name,"model":"tatr","n_tables":r1["n_tables"],"time_sec":r1["time_sec"],"overlay":r1["overlay"]})
            print(f"TATR: tables={r1['n_tables']} time={r1['time_sec']}s")
        except Exception as e:
            print("TATR failed:", e)
            summary_rows.append({"image":img.name,"model":"tatr","error":str(e)})

        # PP-Structure
        try:
            r2 = run_ppstructure(img)
            save_json(r2, OUT_DIR/"ppstructure"/f"{img.stem}.json")
            summary_rows.append({"image":img.name,"model":"ppstructure","n_tables":r2["n_tables"],"time_sec":r2["time_sec"],"overlay":r2["overlay"]})
            print(f"PP-Structure: tables={r2['n_tables']} time={r2['time_sec']}s")
        except Exception as e:
            print("PP-Structure failed:", e)
            summary_rows.append({"image":img.name,"model":"ppstructure","error":str(e)})

        # YOLO (optional)
        try:
            r3 = run_yolo(img)
            save_json(r3, OUT_DIR/"yolo"/f"{img.stem}.json")
            n = r3.get("n_tables", 0)
            t = r3.get("time_sec", None)
            print(f"YOLO: tables={n} time={t}s" if "skipped" not in r3 else "YOLO skipped (no weights)")
            summary_rows.append({"image":img.name,"model":"yolo","n_tables":n,"time_sec":t, "note":r3.get("skipped","")})
        except Exception as e:
            print("YOLO failed:", e)
            summary_rows.append({"image":img.name,"model":"yolo","error":str(e)})

    # write summary
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR/"summary.csv", index=False)
    print("\nDone. Outputs:")
    print(f"- JSON per model: {OUT_DIR}/tatr | ppstructure | yolo")
    print(f"- Overlays next to page images (suffix _overlay.png)")
    print(f"- Summary CSV: {OUT_DIR/'summary.csv'}")

if __name__ == "__main__":
    main()
