import cv2
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

DB_PATH = Path("faces.db")
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MIN_FACE_SIZE = (100, 100)
IMG_SIZE = (200, 200)
UNKNOWN_THRESHOLD = 55.0
VOTE_WINDOW = 8
VOTE_REQUIRED = 5

def compute_person_thresholds(recognizer, persons, faces, labels):
    """Return dict: {person_id: threshold} using μ+2 capped at 70."""
    dists_by_pid = defaultdict(list)
    for img, pid in zip(faces, labels):
        try:
            pred_label, dist = recognizer.predict(img)
        except cv2.error:
            continue
        if pred_label == pid and dist is not None:
            dists_by_pid[pid].append(dist)

    thresholds = {}
    for pid, arr in dists_by_pid.items():
        if len(arr) >= 5:
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            thr = min(mu + 2.0 * sd, 70.0)  # cap to avoid being too loose
            thresholds[pid] = thr
        else:
            thresholds[pid] = UNKNOWN_THRESHOLD  # fallback
    return thresholds


def majority_vote(label_queue):
    counts = defaultdict(int)
    for lab in label_queue:
        counts[lab] += 1
    label, count = max(counts.items(), key=lambda kv: kv[1])
    if label == "Unknown":
        return label if count >= VOTE_REQUIRED else None
    return label if count >= VOTE_REQUIRED else None

def ensure_cv2_face():
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face not found.")

def prepare_face(gray_roi):
    face = cv2.resize(gray_roi, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.equalizeHist(face)

def load_people_and_samples(db_path: Path):
    """Load {id: name} and list of (image, label) for training."""
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT id, name FROM persons")
        persons = {pid: name for pid, name in cur.fetchall()}

        faces = []
        labels = []
        for pid in persons.keys():
            cur.execute("SELECT image_path FROM face_samples WHERE person_id = ?", (pid,))
            for (img_path,) in cur.fetchall():
                p = Path(img_path)
                if not p.exists():
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(prepare_face(img))
                labels.append(pid)

    return persons, faces, labels

def train_lbph(faces, labels):
    ensure_cv2_face()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained = False
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        trained = True
        return recognizer, trained
    return recognizer, False

def predict(recognizer, trained, persons, gray, box, per_person_thr):
    x, y, w, h = box
    roi = prepare_face(gray[y:y+h, x:x+w])
    if not trained:
        return "Unknown", True, None
    label, dist = recognizer.predict(roi)
    thr = per_person_thr.get(label, UNKNOWN_THRESHOLD)
    is_unknown = (label not in persons) or (dist is None) or (dist > thr)
    return (persons.get(label, "Unknown"), is_unknown, dist)

def main():
    persons, faces, labels = load_people_and_samples(DB_PATH)
    recognizer, trained = train_lbph(faces, labels)

    per_person_thr = {}
    if trained:
        per_person_thr = compute_person_thresholds(recognizer, persons, faces, labels)

    cascade = cv2.CascadeClassifier(HAAR_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return
    recent = deque(maxlen=VOTE_WINDOW)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)
        for (x, y, w, h) in detections:
            name, is_unknown, _ = predict(recognizer, trained, persons, gray, (x, y, w, h), per_person_thr)
            label = "Unknown" if is_unknown else name
            recent.append(label)
            voted = majority_vote(recent)
            final_label = voted if voted is not None else label
            color = (0, 0, 255) if is_unknown else (0, 255, 0)
            label = "Unknown" if is_unknown else name

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Multi-person Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
