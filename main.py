import warnings
warnings.filterwarnings("ignore")
import time
from QRec import QRec
from util.config import ModelConf


if __name__ == '__main__':
    s = time.time()
    conf = ModelConf('./config/' + 'MHCN' + '.conf')
    recSys = QRec(conf)
    recSys.train_and_evaluate()
    # recSys.predict()
    e = time.time()
    print("Running time: %f s" % (e - s))
