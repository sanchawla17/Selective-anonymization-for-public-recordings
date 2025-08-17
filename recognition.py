import cv2
import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path("faces.db")
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MIN_FACE_SIZE = (100, 100)
IMG_SIZE = (200, 200)

# LBPH returns a distance (lower is better)
UNKNOWN_THRESHOLD = 75.0

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
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        return recognizer, True
    return recognizer, False

def predict(recognizer, trained, persons, gray, box):
    x, y, w, h = box
    roi = prepare_face(gray[y:y+h, x:x+w])
    if not trained:
        return "Unknown", True, None
    label, dist = recognizer.predict(roi)
    is_unknown = (label not in persons) or (dist is None) or (dist > UNKNOWN_THRESHOLD)
    return (persons.get(label, "Unknown"), is_unknown, dist)

def main():
    persons, faces, labels = load_people_and_samples(DB_PATH)
    recognizer, trained = train_lbph(faces, labels)

    cascade = cv2.CascadeClassifier(HAAR_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)
        for (x, y, w, h) in detections:
            name, is_unknown, _ = predict(recognizer, trained, persons, gray, (x, y, w, h))
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
