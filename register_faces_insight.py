import argparse, os, pickle, cv2
from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis

BANK_DIR = Path("private_facebank")
DB_PATH = BANK_DIR / "encodings_insight.pkl"
IMAGES_DIR = BANK_DIR / "images"

def load_db():
    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)  # {"embeddings":[np.ndarray], "names":[str]}
    return {"embeddings": [], "names": []}

def save_db(db):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def l2_norm(x):
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()

    db = load_db()

    # Init InsightFace (detector + ArcFace)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    captured = 0
    print(f"[INFO] Enrolling {args.name}. Look at the camera. SPACE=capture, Q=quit")
    while captured < args.shots:
        ok, frame = cap.read()
        if not ok: 
            continue
        view = frame.copy()
        cv2.putText(view, f"Enroll {args.name}  ({captured}/{args.shots})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
        cv2.imshow("Register Face", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == 32:  # SPACE KEY
            faces = app.get(frame)
            if len(faces) != 1:
                print(f"[WARN] Need exactly 1 face, found {len(faces)}. Try again.")
                continue
            emb = l2_norm(faces[0].normed_embedding.astype("float32"))
            db["embeddings"].append(emb)
            db["names"].append(args.name)
            captured += 1
            if args.save_crops:
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                person_dir = IMAGES_DIR / args.name
                person_dir.mkdir(parents=True, exist_ok=True)
                box = faces[0].bbox.astype(int)
                crop = frame[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(str(person_dir / f"{args.name}_{captured}.jpg"), crop)
            print(f"[OK] Sample {captured}/{args.shots}")

    cap.release(); cv2.destroyAllWindows()
    if captured:
        save_db(db)
        print(f"[OK] Saved to {DB_PATH}")

if __name__ == "__main__":
    main()