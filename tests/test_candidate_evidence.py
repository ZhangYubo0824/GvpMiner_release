import pytest
from scripts.candidate_evidence.evidence import unique_index


def test_duplicate_candidate_key_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        unique_index([{'id': 'a'}, {'id': 'a'}], 'id')
