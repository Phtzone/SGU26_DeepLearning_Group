"""Window-local preprocessing for a validation-only CNN diagnostic."""
import numpy as np
from src.data.dataset import WindowDataset

# Preserve absolute level and scale of the two EDA and two temperature channels.
OSCILLATORY_CHANNELS = ('chest_ACC_X','chest_ACC_Y','chest_ACC_Z','chest_ECG','chest_EMG',
                       'chest_Resp','wrist_ACC_X','wrist_ACC_Y','wrist_ACC_Z','wrist_BVP')


def window_zscore(window, channels):
    x=np.asarray(window)
    if x.ndim!=2 or x.shape[0]!=len(channels) or not np.isfinite(x).all():
        raise ValueError('Expected finite [channels,time] window.')
    result=x.copy()
    indices=[i for i,c in enumerate(channels) if c in OSCILLATORY_CHANNELS]
    a=x[indices].astype(np.float64)
    std=a.std(axis=1,keepdims=True)
    std[std<1e-8]=1.
    result[indices]=((a-a.mean(axis=1,keepdims=True))/std).astype(x.dtype)
    return result


class PreprocessedWindowDataset(WindowDataset):
    def __init__(self, cache, subjects, channels, mode):
        super().__init__(cache,subjects)
        if not self.timeline or mode not in ('control','window_zscore'):
            raise ValueError('Expected common-rate timelines and supported preprocessing mode.')
        self.channels,self.mode=list(channels),mode

    def raw_window(self,subject,offset):
        x=super().raw_window(subject,offset)
        return window_zscore(x,self.channels) if self.mode=='window_zscore' else x
