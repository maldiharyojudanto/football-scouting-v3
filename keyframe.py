import cv2
import os
import numpy as np
from modules.keyframe_module import cari_hog, keyframe_extraction
import random

# Membaca file video.mp4
VIDEO_PATH = "G:/TA Video/dfl-bundesliga-data-shootout/train/ecf251d4_0.mp4" # Source video dalam lokal
cap = cv2.VideoCapture(VIDEO_PATH) # 0 = webcam, 1 = external webcam, VIDEO_PATH = lokasi video lokal

# Membaca detail frame video
file_base = os.path.basename(VIDEO_PATH) # Get nama direktori file dan nama file
file_name = os.path.splitext(file_base) # Get nama file saja 
frame_jumlah = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get total frame dalam file video masukan
fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS (Frame Per Second) dalam file video masukan
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Get ukuran width dari frame video masukan
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get ukuran height dari frame video masukan

# Menampilkan informasi video
print("============= DETAIL FILE VIDEO =============")
print(f"Nama File Video :", file_name, 
      "\nTotal Frame :", int(frame_jumlah), 
      "\nFPS :", int(fps), 
      "\nDurasi Video (Detik) :", frame_jumlah/fps,
      "\nOriginal Ukuran Frame :", int(frame_width), int(frame_height)) 
print("=============================================")

OUTPUT_PATH_IMAGE = os.path.join("output\images\keyframes", "dfl-bundesliga-data-shootout", file_name[0]) 
try:
  os.makedirs(OUTPUT_PATH_IMAGE)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_IMAGE)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_IMAGE)

#################
# frame_nomor = 0

frames_asli = []
frames = []
frame_awal = random.randint(78001, 80000)
frame_akhir = frame_awal+100
#################

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_awal)

# while cap.isOpened():
while frame_awal < frame_akhir :
  # Membaca frame saat ini yang telah diekstrak
  success, frame = cap.read()

  if success:
    frame_awal += 1
    print(f"Sedang memproses frame ke-{frame_awal}")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konversi frame dari 3 channel ke 1 channel gray
    frame_resize = cv2.resize(frame_gray, (1280,720)) # Resize gambar supaya efisien

    fd, hog_image = cari_hog(frame_resize) # Cari HOG dari frame

    # pca = PCA(n_components=80) # Inisiasi component
    # img_compressed = pca.fit_transform(frame_gray) # Dilakukan fitting
    # img_decompressed = pca.inverse_transform(img_compressed) # Decompresi untuk mengubah ke ukuran asli img_gray

    # frame_reshape = np.reshape(frame_gray, (int(frame_height)* int(frame_width))) # Flattening dari 2D menjadi 1D

    # print(len(frame_reshape))
    # print(len(fd))
    frames_asli.append(frame)
    frames.append(fd) # Ditampung ke array
    print(f"Berhasil memasukkan frame {frame_awal} ke array")

    # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  else:
    # Berhenti ketika sampai frame terakhir
    break
    
cap.release()
cv2.destroyAllWindows()

# keyframe(jumlahcluster_atau_banyakframeyangdiekstrak, frame_heiht, frame_width, direktori_output)
keyframe_extraction(2, frames, frames_asli, OUTPUT_PATH_IMAGE)