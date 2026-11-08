import csv
from datetime import datetime, timedelta
import os
import cv2
import face_recognition
import numpy as np

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "absensi.csv"
LOG_EVAL_FILE = "evaluasi_log.csv"


def init_files():
  if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

  if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["NIM", "NAMA", "WAKTU", "STATUS"])

  if not os.path.exists(LOG_EVAL_FILE):
    with open(LOG_EVAL_FILE, mode="w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow([
          "TIMESTAMP",
          "NIM",
          "NAMA",
          "JARAK_EUCLIDEAN",
          "STATUS_AKURASI",
      ])


def preprocess_image_lab(image):
  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  elif image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

  lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  cl = clahe.apply(l)
  limg = cv2.merge((cl, a, b))
  enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
  return enhanced_rgb


def load_known_faces():
  known_encodings = []
  known_names = []
  known_nims = []

  if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

  files = [
      f
      for f in os.listdir(DATASET_DIR)
      if f.lower().endswith((".jpg", ".jpeg", ".png"))
  ]
  if not files:
    return known_encodings, known_names, known_nims

  for filename in files:
    name_nim = os.path.splitext(filename)[0]
    parts = name_nim.split("_")
    if len(parts) >= 2:
      nim = parts[0]
      name = " ".join(parts[1:])

      image_path = os.path.join(DATASET_DIR, filename)
      image = face_recognition.load_image_file(image_path)

      processed_img = preprocess_image_lab(image)
      encodings = face_recognition.face_encodings(processed_img)

      if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
        known_nims.append(nim)

  return known_encodings, known_names, known_nims


def mark_attendance(nim, name):
  now = datetime.now()
  date_str = now.strftime("%Y-%m-%d")
  time_str = now.strftime("%H:%M:%S")

  already_marked = False
  if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode="r", encoding="utf-8") as f:
      reader = csv.reader(f)
      next(reader, None)
      for row in reader:
        if len(row) >= 3:
          row_nim = row[0]
          row_date = row[2].split(" ")[0]
          if row_nim == nim and row_date == date_str:
            already_marked = True
            break

  if not already_marked:
    with open(ATTENDANCE_FILE, mode="a", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow([nim, name, f"{date_str} {time_str}", "HADIR"])
    print(f"\n[ABSEN BERHASIL] {name} ({nim}) tercatat pada {time_str}")


def log_evaluation_throttled(
    nim, name, distance, last_logged_dict, threshold
):
  now = datetime.now()
  status = "BENAR (MATCH)" if distance < threshold else "SALAH/TIDAK DIKENAL"

  if nim not in last_logged_dict or (now - last_logged_dict[nim]) > timedelta(
      seconds=5
  ):
    last_logged_dict[nim] = now
    with open(LOG_EVAL_FILE, mode="a", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(
          [now.strftime("%Y-%m-%d %H:%M:%S"), nim, name, f"{distance:.4f}", status]
      )


def main():
  init_files()
  known_encodings, known_names, known_nims = load_known_faces()

  video_capture = cv2.VideoCapture(0)
  if not video_capture.isOpened():
    print("[ERROR] Kamera tidak dapat diakses.")
    return

  print("\n" + "=" * 50)
  print(" APLIKASI ABSENSI PENGENALAN WAJAH (PCD 2026)")
  print("=" * 50)
  if not known_encodings:
    print(
        "[INFO] Database kosong. Arahkan wajah ke kamera, klik jendela kamera,"
        " lalu tekan 's'!"
    )
  else:
    print(f"[INFO] Total wajah terdaftar di database: {len(known_encodings)}")
  print("[INFO] Tekan 's' di jendela kamera untuk Mendaftarkan Wajah Baru.")
  print("[INFO] Tekan 'q' untuk Keluar Aplikasi.")
  print("=" * 50 + "\n")

  RECOGNITION_THRESHOLD = 0.45
  last_logged_times = {}

  while True:
    ret, frame = video_capture.read()
    if not ret:
      print("[ERROR] Gagal membaca stream dari kamera.")
      break

    t_start = datetime.now()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    processed_rgb = preprocess_image_lab(rgb_frame)
    small_frame = cv2.resize(processed_rgb, (0, 0), fx=0.5, fy=0.5)

    face_locations = face_recognition.face_locations(
        small_frame, model="hog"
    )
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    t_end = datetime.now()
    processing_time_ms = (t_end - t_start).total_seconds() * 1000

    for face_encoding, face_location in zip(face_encodings, face_locations):
      name = "Tidak Dikenal"
      nim = "UNKNOWN"
      display_distance = 1.0

      if known_encodings:
        face_distances = face_recognition.face_distance(
            known_encodings, face_encoding
        )
        if len(face_distances) > 0:
          best_match_index = np.argmin(face_distances)
          display_distance = face_distances[best_match_index]

          if display_distance < RECOGNITION_THRESHOLD:
            name = known_names[best_match_index]
            nim = known_nims[best_match_index]
            mark_attendance(nim, name)
            log_evaluation_throttled(
                nim,
                name,
                display_distance,
                last_logged_times,
                RECOGNITION_THRESHOLD,
            )

      top, right, bottom, left = [int(coord * 2) for coord in face_location]

      box_color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
      cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
      cv2.rectangle(
          frame, (left, bottom - 45), (right, bottom), box_color, cv2.FILLED
      )

      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(
          frame,
          f"{name}",
          (left + 6, bottom - 25),
          font,
          0.6,
          (255, 255, 255),
          1,
      )
      cv2.putText(
          frame,
          f"Dist: {display_distance:.2f}",
          (left + 6, bottom - 6),
          font,
          0.4,
          (255, 255, 255),
          1,
      )

    cv2.putText(
        frame,
        f"Proc Time: {processing_time_ms:.1f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Aplikasi Absensi Wajah - PCD 2026", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("s"):
      # Tutup jendela OpenCV sementara agar tidak 'Not Responding' saat input terminal
      cv2.destroyAllWindows()

      print("\n--- PENDAFTARAN WAJAH BARU ---")
      input_nim = input("Masukkan NIM: ").strip()
      input_name = input("Masukkan Nama: ").strip()

      if input_nim and input_name:
        filename = f"{input_nim}_{input_name}.jpg"
        filepath = os.path.join(DATASET_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(
            f"[SUKSES] Foto berhasil disimpan ke {filepath}. Memuat ulang"
            " database..."
        )

        known_encodings, known_names, known_nims = load_known_faces()
        print(f"[INFO] Total wajah terdaftar sekarang: {len(known_encodings)}")
      else:
        print("[GAGAL] NIM dan Nama tidak boleh kosong!")
      print("------------------------------\n")

  video_capture.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()