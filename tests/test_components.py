import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EMBED_BACKEND", "hash")      # tests never need the 0.5 GB model

from src.agent.normalize import normalize_asr
from src.data.catalog import Catalog
from src.kb.loader import load_chunks
from src.retrieval.hybrid import HybridRetriever, rrf


def test_normalizer_sku_and_year():
    assert "ele-0012" in normalize_asr("so sku ele-zero zero one to pe assured hai")
    assert "2026" in normalize_asr("refund 17 october two zero two six")
    assert "upi" in normalize_asr("uh u p i wala refund")


def test_normalizer_leaves_normal_to_alone():
    assert normalize_asr("i want to return it") == "i want to return it"


def test_catalog_resolves_sku_and_title():
    c = Catalog()
    assert c.resolve("sku ELE-0012 pe assured?").sku == "ELE-0012"
    assert c.resolve("Pagecraft A5 Ruled Notebook Pack of 4").return_category == "books_media"


def test_chunk_ids_unique_and_nonempty():
    ids = [c.section_id for c in load_chunks()]
    assert len(ids) == len(set(ids)) and len(ids) > 50


def test_rrf_prefers_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert max(fused, key=fused.get) == 1


def test_hybrid_returns_k_hits_with_ranks():
    r = HybridRetriever(load_chunks())
    hits = r.search("how long do refunds to upi take", k=5)
    assert len(hits) == 5 and hits[0].chunk.section_id
