import cv2
import os

# Membaca file video.mp4
VIDEO_PATH = "G:/TA Video/path/to/SoccerNet/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/1_720p.mkv" # Source video dalam lokal
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

# Simpan video ke ukuran 1280x720 (supaya lebih efisien & kompresi ukuran file)
NEW_FRAME_WIDTH = 1280
NEW_FRAME_HEIGHT = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Format video
OUTPUT_PATH_FOLDER = os.path.join("output/videos/sbd/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea") # Lokasi simpan video hasil inference
# OUTPUT_PATH_IMAGES = os.path.join("output\images\sbd", "dfl-bundesliga-data-shootout", file_name[0]) 
try:
# os.makedirs(OUTPUT_PATH_IMAGES)
  os.makedirs(OUTPUT_PATH_FOLDER)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_FOLDER)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_FOLDER)
OUTPUT_PATH_VIDEOS = OUTPUT_PATH_FOLDER+f"/{file_name[0]} output.mp4"
out = cv2.VideoWriter(OUTPUT_PATH_VIDEOS, fourcc, fps, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT)) # Untuk simpan

#################
frame_awal = 0

ret, frame_sebelum = cap.read()
frame_sebelum_gray = cv2.cvtColor(frame_sebelum, cv2.COLOR_BGR2GRAY) # Konversi frame dari 3 channel ke 1 channel gray

cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

# frame_sbd = []
treshold = 11

# print(frame_sebelum_gray)
#################

# while cap.isOpened():
while frame_awal < frame_jumlah:
    # Membaca frame saat ini yang telah diekstrak
    success, frame_sekarang = cap.read()

    if success:
        frame_awal += 1
        print(f"Sedang memproses frame ke-{frame_awal}")
        frame_sekarang_gray = cv2.cvtColor(frame_sekarang, cv2.COLOR_BGR2GRAY)
        # print(frame_sekarang_gray)
        frame_perbedaan = cv2.absdiff(frame_sebelum_gray, frame_sekarang_gray) # Menghitung perbedaan antara frame sebelum dan sekarang
        frame_perbedaan_rata = frame_perbedaan.mean() # Mencari mean dari perbedaan frame

        print(frame_perbedaan_rata, treshold)

        if frame_perbedaan_rata > treshold: # Jika perbedaan lebih besar daripada treshold maka 
            # frame_sbd.append(frame_sekarang_gray)
            # print(f"Berhasil memasukkan frame {frame_awal} ke array")

            # cv2.imwrite(OUTPUT_PATH_IMAGES+"/frame %d.jpg" % frame_awal, frame_sekarang) # Jika ingin menyimpan image
            # print(f"Sedang menyimpan frame ke-{frame_awal} ke lokal storage")

            # Simpan/add per frame ke format video
            out.write(frame_sekarang)

            # # Tampilkan hasil di layar
            # cv2.imshow("{} Tracking".format(file_name[0]), frame_sekarang)
            # # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        frame_sebelum_gray = frame_sekarang_gray # Set frame prev menjadi frame n-1
    else:
        # Berhenti ketika sampai frame terakhir
        break

# Release the video capture object and close the display window
print("\nOutput video telah berhasil disimpan pada '{}!'".format(OUTPUT_PATH_VIDEOS))
cap.release()
cv2.destroyAllWindows()

# for i, frame in enumerate(frame_sbd):
#     # out.write(frame)
    
# out.release()
