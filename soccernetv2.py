import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

# Lokasi simpan
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="G:/TA Video/path/to/SoccerNet")

# Download data, password = "s0cc3rn3t"
mySoccerNetDownloader.password = input("Password for videos (received after filling the NDA) ")
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])