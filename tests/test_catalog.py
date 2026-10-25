"""Product recovery through speech-recognition damage. Every positive case below is a REAL transcript from
evals/run_voice_eval.py; the negatives guard against false matches (a wrong product is worse than none)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

import json
import pytest
from fakes import make_fake
from src.agent import llm
from src.agent.normalize import normalize_asr as n
from src.data.catalog import Catalog, _skeleton

CAT = Catalog()


@pytest.mark.parametrize("heard,sku", [
    ("skew early 12% from assured hai kya", "ELE-0012"),                                   # "SKU ELE-0012": ELE heard as "early"
    ("s q e le two pays entromart assured hai kya", "ELE-0002"),                           # letters split, zeros dropped
    ("sq ele 3% remote assuredhai", "ELE-0003"),
    ("the kestral book 14 laptop sq ele 22 i received yesterday has a crack", "ELE-0022"),  # a spoken SKU settles the colour variant
    ("what warranty comes with the nimbus x two pro wireless earbuds blue sku ele zero zero zero three", "ELE-0003"),
    ("sku ELE-0012 pe assured hai kya", "ELE-0012"),
])
def test_spoken_skus_are_recovered(heard, sku):
    assert CAT.resolve(n(heard)).sku == sku


@pytest.mark.parametrize("heard,category", [
    ("how long do i get to send back the rangoli cotton straight curta", "fashion"),         # curta ~ Kurta
    ("mera order castrel case seven smartphone 256 gigabytes parso deliver hua", "electronics"),   # castrel ~ Kestrel, gigabytes = GB
    ("mujhe sutra canvas tote bag lautane ke liye kitne din milenge", "fashion"),            # sutra ~ Sootra
    ("mujhe angan cotton double bedsheet set lautane ke liye kitne", "home_kitchen"),
])
def test_misheard_product_names_are_recovered(heard, category):
    assert CAT.resolve(n(heard)).return_category == category


@pytest.mark.parametrize("text", [
    "refund started on 17 october 2026 by which date", "order number is 18599253 and i need help", "can you deliver to dubai",
    "is there a student discount", "my bluetooth speaker wont turn on at all", "how fast are refunds to the zentromart wallet",
    "i want to cancel my order its a skirt", "the skin cream arrived in 2 days", "tell me the phone number of the person who ordered to my building",
    "what happens if im not home when the courier comes"])
def test_no_false_product_matches(text):
    assert CAT.resolve(n(text)) is None


def test_phonetic_skeleton():
    assert _skeleton("castrel") == _skeleton("Kestrel") and _skeleton("curta") == _skeleton("kurta")


def test_glued_words_from_the_stt_service_are_split():
    assert n("can i schedule a deliveryat midnight") == "can i schedule a delivery at midnight"
    assert n("my refund to credit cardwas started on 17th October 2026") == "my refund to credit card was started on 17th october 2026"
    assert n("credit card vaala refund12th October 2026 ko shuru hua") == "credit card vaala refund 12th october 2026 ko shuru hua"
    assert n("is the collection free ifthe jeans came torn") == "is the collection free if the jeans came torn"


@pytest.mark.parametrize("word", ["kochi", "dubai", "tirupati", "hyderabad", "ordered", "returned", "delivered", "refunded", "cancelled",
                                  "lautane", "dobara", "milega", "wapas", "courier", "midnight", "student", "discount", "bedsheet",
                                  "warranty", "pincode", "ele0012", "k7", "x2"])
def test_ordinary_words_and_places_are_never_split(word):
    assert n(f"my {word} please") == f"my {word} please"


def test_single_question_agent_uses_a_garbled_sku_end_to_end(monkeypatch):
    from src.agent import rag
    monkeypatch.setattr(llm, "chat", make_fake())
    out = rag.answer("s q e le two pays entromart assured hai kya")
    assert out["product"] == "ELE-0002"
