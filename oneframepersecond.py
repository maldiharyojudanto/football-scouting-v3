import cv2
import os

# Membaca file video.mp4
VIDEO_PATH = "output/videos/sbd/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/1_720p output.mp4" # Source video dalam lokal
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

OUTPUT_PATH_IMAGE = os.path.join("output\images\sbd", "SoccerNet/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea", file_name[0]) 
try:
  os.makedirs(OUTPUT_PATH_IMAGE)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_IMAGE)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_IMAGE)

#################
frame_nomor = 0
#################

while cap.isOpened():
  # Membaca frame saat ini yang telah diekstrak
  success, frame = cap.read()

  if success:
    frame_nomor += 1
    print(f"Sedang memproses frame ke-{frame_nomor}")
    
    if (frame_nomor % int(fps) == 0): # Jika frame nomor di mod dengan fps hasilnya 0 maka
        cv2.imwrite(OUTPUT_PATH_IMAGE+"/frame %d.jpg" % frame_nomor, frame) # Jika ingin menyimpan image
        print(f"Sedang menyimpan frame ke-{frame_nomor} ke lokal storage")

    # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  else:
    # Berhenti ketika sampai frame terakhir
    break
    
cap.release()
cv2.destroyAllWindows()