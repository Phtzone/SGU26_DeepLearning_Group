import numpy as np
import pytest
from src.data.window_preprocessing import window_zscore


def test_center_scale_only_selected_channels_and_preserve_original():
    channels=['chest_ECG','wrist_EDA','chest_Temp']
    x=np.array([[1,2,3,4],[10,11,12,13],[33,33,34,34]],dtype=np.float32);old=x.copy()
    z=window_zscore(x,channels)
    np.testing.assert_allclose(z[0].mean(),0,atol=1e-7)
    np.testing.assert_allclose(z[0].std(),1,atol=1e-7)
    assert np.array_equal(z[1:],x[1:]) and np.array_equal(x,old)


def test_constant_channel_finite_and_no_other_window_dependency():
    x=np.ones((2,64),dtype=np.float32)*7
    z=window_zscore(x,['chest_ACC_X','wrist_TEMP'])
    assert np.isfinite(z).all() and (z[0]==0).all() and (z[1]==7).all()
    other=window_zscore(x*100,['chest_ACC_X','wrist_TEMP'])
    assert (z[0]==other[0]).all()


def test_nonfinite_input_fails():
    with pytest.raises(ValueError):window_zscore(np.array([[float('nan')]]),['chest_ECG'])


def test_dataset_scaler_fits_transformed_train_and_applies_once(tmp_path):
    from src.data.window_preprocessing import PreprocessedWindowDataset
    from src.data.normalization import ChannelScaler
    from src.utils.tracking import write_json
    import csv
    channels=['chest_ECG','wrist_EDA','wrist_TEMP']
    source=tmp_path/'source';source.mkdir()
    cache=tmp_path/'index';cache.mkdir()
    for subject,shift in [('S2',0),('S3',100000)]:
        raw=np.stack([np.arange(128)+shift,np.arange(128)+10+shift,np.ones(128)*33],axis=1).astype(np.float32)
        np.save(source/f'{subject}_X.npy',raw)
        p=cache/subject;p.mkdir()
        write_json(p/'metadata.json',{'cache_key':subject})
        with (p/'windows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['subject_id','window_id','label','start_index','end_index']);w.writeheader()
            w.writerows({'subject_id':subject,'window_id':f'{subject}_{i}','label':i,'start_index':i*64,'end_index':(i+1)*64} for i in range(2))
    write_json(cache/'common_rate_index.json',{'audit':{'source':str(source),'notebook_config':{'channels':channels},'policy':{'channels':channels}}})
    train=PreprocessedWindowDataset(cache,['S2'],channels,'window_zscore')
    val=PreprocessedWindowDataset(cache,['S3'],channels,'window_zscore')
    scaler=ChannelScaler().fit(train)
    assert scaler.subjects==['S2']
    np.testing.assert_allclose(scaler.mean,[0,73.5,33],atol=1e-6)
    np.testing.assert_allclose(scaler.std[0],1,atol=1e-6)
    expected=scaler.transform(train.raw_window('S2',0))
    train.scaler=scaler
    assert np.array_equal(train[0][0].numpy(),expected)
    assert val.raw_window('S3',0)[1].mean()>100000
