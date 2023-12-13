import cv2
import os
import numpy as np
# from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.cluster import KMeans

def cari_hog(frame):
    # Cari HOG dari frame
    fd, hog_image = hog(frame, orientations=9, pixels_per_cell=(12, 12),
                	cells_per_block=(2, 2), visualize=True)
    
    return fd, hog_image

def keyframe_extraction(k, frames, frame_asli, OUTPUT_PATH_IMAGE):
    frames_array = np.vstack(frames) # Array frames ditumpuk secara vertikal
    print(frames_array)

    print("Sedang Training ... :)")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto") # Dilakukan clustering
    kmeans.fit(frames_array)
    print("Selesai Clustering ... :)")

    labels = kmeans.labels_ # Dapatkan labels hasil clustering/prediksi setiap row pada array
    centroid = kmeans.cluster_centers_ # Dapatkan titik tengah/centroid

    rata_frames_array = frames_array.mean(axis=1) # Mencari rata setiap row pada array
    print(rata_frames_array)

    frame_idx = []
    for i in centroid: # centroid = n_cluster
        jumlah = i.sum() # Menghitung atau menjumlahkan setiap element pada element i
        mean = jumlah/len(i) # Mencari rata-rata dari penjumhlahan element i/panjang i
        print(mean)
        jarak = np.abs(rata_frames_array - mean) # Jarak antara setiap element array dengan mean
        print(jarak)
        index_terdekat = np.argmin(jarak) # Cari nilai yang mendekati 0 (terdekat)
        frame_idx.append(index_terdekat)

    for i in frame_idx:
        # frame_asli = np.reshape(frames_array[i], (int(frame_height), int(frame_width))) # Deresize menjadi ukuran awal (2D)
        # frame_rgb = cv2.cvtColor(frame_asli[i], cv2.COLOR_GRAY2BGR)
        file_path = OUTPUT_PATH_IMAGE+f"/frame {i+180} {k}.jpg"

        if os.path.exists(file_path):
            cv2.imwrite(OUTPUT_PATH_IMAGE+f"/frame {i+180} {k}.jpg", frame_asli[i]) # save frame ke lokal
        else:
            cv2.imwrite(OUTPUT_PATH_IMAGE+f"/frame {i+180} {k}.jpg", frame_asli[i]) # save frame ke lokal
        print(f"Berhasil memasukkan frame {i+180} ke lokal storage")