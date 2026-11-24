from copy import deepcopy
import pytest
from scripts.finish_common_rate_gru import paired_metrics


def rows():
    return [{'subject_id':s,'window_id':f'{s}_{i}','true_label':str(i),
             'predicted_label':str(i),'fold_id':'1','start_time':str(i*15),'end_time':str(i*15+30)}
            for s in ('S2','S3') for i in range(3)]


def test_primary_metric_is_equal_weight_per_person():
    cnn=rows();gru=deepcopy(cnn)
    for r in gru:
        if r['subject_id']=='S3':r['predicted_label']=str((int(r['true_label'])+1)%3)
    paired,summary=paired_metrics(cnn,gru)
    assert summary['CNN']['macro_f1']==1
    assert summary['GRU']['macro_f1']==.5
    assert [r['difference_gru_minus_cnn'] for r in paired]==[0,-1]


@pytest.mark.parametrize('change',['duplicate','id','label','person','fold','time'])
def test_incompatible_comparisons_fail(change):
    cnn=rows();gru=deepcopy(cnn)
    if change=='duplicate':gru.append(dict(gru[0]))
    elif change=='id':gru[0]['window_id']='different'
    else:gru[0][{'label':'true_label','person':'subject_id','fold':'fold_id','time':'end_time'}[change]]='99'
    with pytest.raises(ValueError):paired_metrics(cnn,gru)
