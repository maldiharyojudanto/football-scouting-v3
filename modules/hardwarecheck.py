import torch

# Cek informasi GPU dan CPU Hardware
def cudagpu():
    print("============== DETAIL HARDWARE ==============")
    print(f"Versi Pytorch Terinstal :",torch.__version__,
        "\nApakah GPU Tersedia? ", torch.cuda.is_available(),
        "\nNama Device :", torch.cuda.get_device_name(0))
    print("=============================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device