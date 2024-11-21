import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/root/codes/soccernet/videos/train")

pswd = 's0cc3rn3t'
mySoccerNetDownloader.password = pswd
mySoccerNetDownloader.downloadGames(files=["Labels-caption.json"], split=["train","valid","test","challenge"])
