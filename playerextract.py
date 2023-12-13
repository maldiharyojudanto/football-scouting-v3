from ultralytics import YOLO
from modules.hardwarecheck import cudagpu

import cv2
import os

cudagpu()

# Load model
model = YOLO('weights/football-scouting-best-m-1695-aug-segonlyplayer.pt')
# results = model('sample_2.jpg', show=True, save=True) # Contoh prediksi pada single image

# Membaca folder .jpg
IMAGE_PATH = "Football-Scouting-5/train/images" # Source image dalam lokal
OUTPUT_PATH_IMAGE = os.path.join("output\images\playerextract", "1695") 
try:
  os.makedirs(OUTPUT_PATH_IMAGE)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_IMAGE)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_IMAGE)

#################
frame_nomor = 0
#################

for filename in os.listdir(IMAGE_PATH):
    frame_nomor += 1
    frame = cv2.imread(os.path.join(IMAGE_PATH, filename)) # Read gambar pada direktori IMAGE_PATH/filename

    # Prediksi frame
    results = model(frame, imgsz=1280)
    print(f"Sedang memproses frame ke {frame_nomor}")

    # Menampilkan hasil prediksi (per bunding box) pada frame ini
    for r in results:
        boxes = r.boxes
        box_nomor = 0
        for box in boxes:
            box_nomor += 1
            # print(box)
            # print(box.cls[0])
            kelas = int(box.cls[0])
            # print(kelas)
            # print(box.xyxy[0])
            # print(x1,y1,x2,y2)
            # print(x1,y1,x2,y2)
            x1, y1, x2, y2 = box.xyxy[0] # Mencari titik x1,y1 dan x2,y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if kelas == 2: # Jika kelas ini player
                file_path = OUTPUT_PATH_IMAGE+f"/playerextract {frame_nomor} {box_nomor}.jpg"
                if os.path.exists(file_path):
                    cv2.imwrite(file_path, frame[y1:y2, x1:x2]) # save frame ke lokal
                else:
                    cv2.imwrite(file_path, frame[y1:y2, x1:x2]) # save frame ke lokal
                print(f"Berhasil memasukkan playerextract {frame_nomor} {box_nomor} ke lokal storage")
            else:
                print("Terdeteksi objek tetapi bukan player")