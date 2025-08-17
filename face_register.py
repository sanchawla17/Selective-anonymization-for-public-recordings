import cv2
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("face_dataset")       
DB_PATH = Path("faces.db")            
SAMPLES_PER_PERSON = 30                
MIN_FACE_SIZE = (100, 100)             
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                image_path TEXT,
                captured_at TEXT,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
        """)
        con.commit()

def insert_person(name):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO persons (name, created_at)
            VALUES (?, ?)
        """, (name, datetime.now().isoformat()))
        con.commit()
        return cur.lastrowid  

def insert_face_sample(person_id, image_path):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO face_samples (person_id, image_path, captured_at)
            VALUES (?, ?, ?)
        """, (person_id, str(image_path), datetime.now().isoformat()))
        con.commit()

def capture_samples(person_id, person_name, samples=SAMPLES_PER_PERSON):
    cascade = cv2.CascadeClassifier(HAAR_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    person_dir = DATA_DIR / f"{person_name}_{person_id}"
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Capturing {samples} face samples for {person_name}. Press 'q' to stop.")
    captured = 0
    last_saved_ts = 0

    while captured < samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            now = time.time()
            if now - last_saved_ts > 0.2: 
                crop = frame[y:y+h, x:x+w]
                filename = f"{captured+1:03d}.jpg"
                out_path = person_dir / filename
                cv2.imwrite(str(out_path), crop)

                insert_face_sample(person_id, out_path)

                captured += 1
                last_saved_ts = now
                cv2.putText(frame, f"Captured {captured}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Captured {captured} samples for {person_name} into {person_dir}")

def list_persons():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, name, created_at FROM persons ORDER BY created_at DESC")
        rows = cur.fetchall()
    if not rows:
        print("No persons enrolled yet.")
        return
    print("\n--- Enrolled Persons ---")
    for r in rows:
        print(f"ID={r[0]} | Name={r[1]} | Created={r[2]}")

def main():
    init_db()
    print("""
=============================
 Face Registration Utility
=============================
1) Enroll new person
2) List enrolled persons
q) Quit
""")
    while True:
        choice = input("Choose an option [1/2/q]: ").strip().lower()
        if choice == '1':
            name = input("Enter name: ").strip()
            if not name:
                print("Name is required.")
                continue
            person_id = insert_person(name)
            capture_samples(person_id, name, SAMPLES_PER_PERSON)
        elif choice == '2':
            list_persons()
        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()
