import pytest
from src.evaluation.history import parse_value, read_csv


@pytest.mark.parametrize('text,expected', [('147',147),('147.0',147),('147.000',147),
    ('1.47e2',147),('9007199254740993.0',9007199254740993)])
def test_exact_integer_csv_representations(text,expected):
    result=parse_value('windows',text)
    assert result==expected and isinstance(result,int)


@pytest.mark.parametrize('text',['147.5','NaN','inf','-Infinity','invalid'])
def test_fractional_or_nonfinite_counts_are_not_silently_truncated(text):
    with pytest.raises(ValueError):parse_value('windows',text)


def test_read_csv_supports_editor_formatted_count_without_rewriting_file(tmp_path):
    p=tmp_path/'subject_metrics.csv'
    original=b'subject_id,windows,macro_f1\nS10,147.0,0.25\n'
    p.write_bytes(original)
    assert read_csv(p)==[{'subject_id':'S10','windows':147,'macro_f1':.25}]
    assert p.read_bytes()==original
