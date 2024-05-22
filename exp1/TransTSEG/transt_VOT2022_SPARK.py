import sys
import os
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2022.transt_seg_class_SPARK import run_vot_exp

net_path = os.path.join(env_path, 'models/TransTsegm_ep0075.pth.tar')
save_root = '/mnt/TransT/vis'

run_vot_exp('transt', window=0.59, penalty_k=0.265, mask_threshold=0.5, net_path=net_path, save_root=save_root, VIS=False)