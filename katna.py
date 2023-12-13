from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os

# For windows, the below if condition is must.
if __name__ == "__main__":
  vd = Video()   # modul video/API  
  jumlah_ekstrak_gambar = 50 # Jumlah image yang akan diambil
  diskwriter = KeyFrameDiskWriter(location="output/images") # Lokasi image akan disimpan

  VIDEO_PATH = "input/1_720p.mkv" # Source video dalam lokal
  vd.extract_video_keyframes(  # Ekstrak keyframe dan proses data dengan diskwriter
       no_of_frames=jumlah_ekstrak_gambar, file_path=VIDEO_PATH,
       writer=diskwriter
  )