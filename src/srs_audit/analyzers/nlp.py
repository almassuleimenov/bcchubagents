"""Русский NLP-слой: razdel (токены/предложения) + pymorphy3 (леммы, граммемы)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from razdel import sentenize, tokenize


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@dataclass(frozen=True)
class Tok:
    start: int
    stop: int
    text: str
    lemma: str
    tag: str


@lru_cache(maxsize=100_000)
def parse_word(word: str) -> tuple[str, str]:
    w = word.lower().replace("ё", "е")
    if not any(ch.isalpha() for ch in w):
        return w, "PNCT" if not any(ch.isdigit() for ch in w) else "NUMB"
    p = _morph().parse(w)[0]
    return p.normal_form.replace("ё", "е"), str(p.tag)


def tokens(text: str) -> list[Tok]:
    out = []
    for t in tokenize(text):
        lemma, tag = parse_word(t.text)
        out.append(Tok(t.start, t.stop, t.text, lemma, tag))
    return out


def sentences(text: str) -> list[tuple[int, int, str]]:
    return [(s.start, s.stop, s.text) for s in sentenize(text)]


NON_PASSIVE_REFLEXIVES = {
    "появляться", "являться", "оставаться", "остаться", "становиться", "стать", "находиться", "казаться",
    "случаться", "стремиться", "бояться", "нравиться", "учиться", "пытаться", "стараться", "получиться",
    "удаваться", "заканчиваться", "начинаться", "улыбаться", "смеяться", "надеяться", "трудиться",
}


@lru_cache(maxsize=10_000)
def is_transitive_base(reflexive_lemma: str) -> bool:
    """«определяться» → «определять» переходный ⇒ возвратная форма = пассив."""
    if reflexive_lemma in NON_PASSIVE_REFLEXIVES or not reflexive_lemma.endswith(("ся", "сь")):
        return False
    base = reflexive_lemma[:-2]
    for p in _morph().parse(base):
        if p.normal_form == base and "INFN" in p.tag and "tran" in p.tag:
            return True
    return False


def is_passive(tok: Tok) -> bool:
    if "PRTS" in tok.tag and "pssv" in tok.tag:
        return True
    if "PRTF" in tok.tag and "pssv" in tok.tag:
        return False  # причастие-определение («накопленная сумма») — не пассивная конструкция
    if tok.tag.startswith("VERB") and tok.text.lower().endswith(("ся", "сь")) and "3per" in tok.tag:
        return is_transitive_base(tok.lemma)
    return False


def context_quote(text: str, start: int, end: int, max_len: int = 180) -> str:
    """Дословный фрагмент вокруг совпадения: предложение целиком или окно по словам."""
    for s_start, s_stop, s_text in sentences(text):
        if s_start <= start < s_stop:
            if len(s_text) <= max_len:
                return s_text.strip()
            break
    left = max(0, start - max_len // 2)
    right = min(len(text), end + max_len // 2)
    while left > 0 and not text[left - 1].isspace():
        left += 1
        if left >= start:
            break
    while right < len(text) and not text[right].isspace():
        right -= 1
        if right <= end:
            right = end
            break
    return text[left:right].strip()
