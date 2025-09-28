import argparse, pickle, cv2
from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis

BANK_DIR = Path("private_facebank")
DB_PATH = BANK_DIR / "encodings_insight.pkl"

def load_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Facebank not found: {DB_PATH}. Run register script first.")
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)  # {"embeddings": [np.ndarray], "names": [str]}
    embs = np.vstack(db["embeddings"]).astype("float32") if db["embeddings"] else np.zeros((0,512), dtype="float32")
    names = np.array(db["names"], dtype=object)
    # L2-normalize once for cosine
    embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12, None)
    return embs, names

def cosine_similarity(a, b):  # a: (N,512), b: (512,)
    # Both inputs expected normalized
    return (a @ b)

def pixelate(img, block=10):
    h, w = img.shape[:2]
    if h == 0 or w == 0: return img
    small = cv2.resize(img, (max(1,w//block), max(1,h//block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def gauss_blur(img, ksize=35):
    k = max(3, int(ksize) | 1) 
    return cv2.GaussianBlur(img, (k, k), 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--sim-thresh", type=float, default=0.60, help="cosine similarity threshold; higher = stricter match")
    ap.add_argument("--det-width", type=int, default=640, help="detector width (speed/quality tradeoff)")
    ap.add_argument("--blur", choices=["pixelate","gauss"], default="pixelate")
    ap.add_argument("--pixel", type=int, default=10, help="block size for pixelation")
    ap.add_argument("--ksize", type=int, default=35, help="kernel size for gaussian blur")
    args = ap.parse_args()

    known_embs, known_names = load_db()
    print(f"[INFO] Loaded {len(known_names)} samples from {DB_PATH}")

    # Initialize InsightFace
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_width, args.det_width))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        faces = app.get(frame)
        do_blur = (gauss_blur if args.blur == "gauss" else pixelate)

        for f in faces:
            box = f.bbox.astype(int)  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            name = "Unknown"

            if known_embs.shape[0] > 0:
                emb = f.normed_embedding.astype("float32")  
                sims = cosine_similarity(known_embs, emb) 
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= args.sim_thresh:
                    name = str(known_names[best_idx])

            if name == "Unknown":
                roi = frame[y1:y2, x1:x2]
                if roi.size:
                    if args.blur == "gauss":
                        frame[y1:y2, x1:x2] = gauss_blur(roi, args.ksize)
                    else:
                        frame[y1:y2, x1:x2] = pixelate(roi, args.pixel)
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = name if name != "Unknown" else "Unknown"
            y = y1 - 8 if y1 - 8 > 12 else y1 + 18
            cv2.rectangle(frame, (x1, y-20), (x2, y), color, -1)
            cv2.putText(frame, tag, (x1 + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"q=quit  thresh={args.sim_thresh:.2f}  blur={args.blur}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Selective Anonymization (InsightFace)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()