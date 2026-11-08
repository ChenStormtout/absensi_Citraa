import os
import face_recognition
import numpy as np

# Struktur folder uji yang harus disiapkan:
# test_dataset/
#   ├── 23541001/  (nama folder sesuai dengan NIM mahasiswa)
#   │     ├── uji_1.jpg
#   │     └── uji_2.jpg
#   └── 23541002/
#         ├── uji_1.jpg
#         └── uji_2.jpg

TEST_DIR = "test_dataset"
DATASET_DIR = "dataset"


def run_quantitative_evaluation():
  print(
      "[INFO] Menjalankan Pengujian Kuantitatif Biometrik Wajah (PCD 2026)..."
  )

  # 1. Memuat database referensi
  known_encodings = []
  known_names = []
  known_nims = []

  for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
      name_nim = os.path.splitext(filename)[0]
      parts = name_nim.split("_")
      if len(parts) >= 2:
        nim = parts[0]
        name = " ".join(parts[1:])
        img_path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(img)
        if encs:
          known_encodings.append(encs[0])
          known_names.append(name)
          known_nims.append(nim)

  if not known_encodings:
    print("[ERROR] Database referensi kosong. Harap isi folder dataset/.")
    return

  # 2. Pengujian variasi threshold untuk analisis performa (ROC / EER)
  thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

  if not os.path.exists(TEST_DIR):
    print(
        f"[WARNING] Folder '{TEST_DIR}' belum ada. Buat folder tersebut dan"
        " kelompokkan citra uji berdasarkan NIM."
    )
    return

  for th in thresholds:
    total_test = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    print(f"\n========================================")
    print(f" Evaluasi Menggunakan Threshold (Th): {th}")
    print(f"========================================")

    for person_folder in os.listdir(TEST_DIR):
      folder_path = os.path.join(TEST_DIR, person_folder)
      if not os.path.isdir(folder_path):
        continue

      expected_nim = person_folder  # Nama folder merepresentasikan NIM target

      for test_file in os.listdir(folder_path):
        if test_file.lower().endswith((".jpg", ".jpeg", ".png")):
          total_test += 1
          test_img_path = os.path.join(folder_path, test_file)
          test_img = face_recognition.load_image_file(test_img_path)

          test_encs = face_recognition.face_encodings(test_img)
          if not test_encs:
            # Wajah gagal dideteksi oleh model (False Negative / False Rejection)
            false_negative += 1
            continue

          test_encoding = test_encs[0]
          distances = face_recognition.face_distance(
              known_encodings, test_encoding
          )

          if len(distances) > 0:
            best_idx = np.argmin(distances)
            min_dist = distances[best_idx]
            predicted_nim = known_nims[best_idx]

            if min_dist < th:
              if predicted_nim == expected_nim:
                true_positive += 1
              else:
                false_positive += 1  # Salah mengenali identitas lain (False Acceptance)
            else:
              false_negative += 1  # Gagal mengenali (False Rejection)

    # Perhitungan Metrik Standar Biometrik dengan proteksi division-by-zero
    recognition_rate = (
        (true_positive) / total_test * 100 if total_test > 0 else 0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    frr = (
        false_negative / total_test * 100 if total_test > 0 else 0
    )  # False Rejection Rate

    print(f"Total Citra Uji     : {total_test}")
    print(f"True Positive       : {true_positive}")
    print(f"False Positive      : {false_positive}")
    print(f"False Negative      : {false_negative}")
    print(f"Recognition Rate    : {recognition_rate:.2f}%")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print(f"Estimated FRR       : {frr:.2f}%")


if __name__ == "__main__":
  run_quantitative_evaluation()