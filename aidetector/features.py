"""
Извлечение признаков, по которым отличают ИИ-текст от человеческого.

Главная идея: модель генерирует текст, выбирая на каждом шаге наиболее
вероятный токен. Поэтому ИИ-текст в среднем:
  - более предсказуем для языковой модели  -> низкая перплексия
  - РОВНЕЕ по предсказуемости от предложения к предложению -> низкая дисперсия
    перплексии (человек скачет: то банальная фраза, то неожиданная)
  - реже использует «удивительные» токены     -> ранги токенов концентрируются
    в топе распределения -> низкая энтропия ранга
  - более ровный по ритму                  -> низкая burstiness
  - беднее по неожиданной лексике
  - чаще содержит шаблонные связки

Мы НЕ судим по одному признаку. Мы считаем несколько и отдаём их классификатору,
который сам выучит веса.

ВАЖНО про производительность: перплексию (среднее), дисперсию перплексии по
предложениям и энтропию ранга считаем за ОДИН forward-pass на текст
(см. _lm_token_stats) — отдельный проход на каждое предложение был бы в разы
дороже и упёрся бы в CPU.
"""

import math
import re
from dataclasses import dataclass, asdict

import numpy as np
import torch
from razdel import sentenize, tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

# Для русского текста берём русскоязычную GPT, иначе перплексия врёт.
# Для английского замени на "gpt2".
PERPLEXITY_MODEL = "ai-forever/rugpt3small_based_on_gpt2"

# Связки/штампы, которыми злоупотребляют генераторы (рус.).
CLICHES = [
    "важно отметить", "стоит отметить", "таким образом", "в заключение",
    "играет важную роль", "является неотъемлемой частью", "следует подчеркнуть",
    "в современном мире", "не секрет, что", "в первую очередь",
    "необходимо учитывать", "в целом можно сказать", "подводя итог",
]

# Границы корзин для ранга истинного токена в распределении модели (GLTR-подход):
# top-1, top-10, top-100, top-1000, дальше. По долям токенов в этих корзинах
# считаем энтропию — у людей «удивительных» токенов больше, распределение по
# корзинам ровнее -> энтропия выше.
RANK_BINS = [1, 10, 100, 1000]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok = None
_lm = None


def _lazy_load():
    """Грузим тяжёлую модель один раз, лениво — чтобы импорт был быстрым."""
    global _tok, _lm
    if _lm is None:
        # use_fast=True нужен для offset_mapping (привязка токенов к предложениям)
        _tok = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL, use_fast=True)
        _lm = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL).to(_device)
        _lm.eval()


@dataclass
class Features:
    perplexity: float          # средняя "удивлённость" языковой модели
    perplexity_std: float      # разброс перплексии ПО ПРЕДЛОЖЕНИЯМ (у людей выше)
    rank_entropy: float        # энтропия ранга токенов (GLTR): у людей выше
    burstiness: float          # разброс длин предложений (std)
    mean_sent_len: float       # средняя длина предложения в словах
    mattr: float               # лексическое разнообразие, НЕ зависящее от длины
    cliche_rate: float         # доля штампов на 100 слов
    punct_variety: float       # разнообразие пунктуации (норм.)


@torch.no_grad()
def _lm_token_stats(text: str):
    """
    ОДИН forward-pass модели -> сразу три LM-признака.

    Возвращает (perplexity, perplexity_std, rank_entropy) или None, если текста
    мало (<2 токенов).

    Как считаем:
      - per-token loss = -log p(токен) -> perplexity = exp(mean loss).
      - per-token rank = сколько токенов словаря вероятнее истинного. Раскладываем
        ранги по корзинам RANK_BINS и берём энтропию долей -> rank_entropy.
      - привязываем каждый токен к предложению по offset_mapping, считаем
        перплексию каждого предложения и берём std по предложениям -> perplexity_std.
    """
    _lazy_load()
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=1024,
               return_offsets_mapping=True)
    input_ids = enc.input_ids.to(_device)
    if input_ids.size(1) < 2:
        return None
    offsets = enc["offset_mapping"][0].tolist()   # (start, end) символов на токен

    logits = _lm(input_ids).logits[0]             # (T, V)
    # сдвиг: токен t предсказываем из позиций <t
    shift_logits = logits[:-1, :]                 # (T-1, V)
    shift_labels = input_ids[0, 1:]               # (T-1,)
    L = shift_labels.size(0)
    idx = torch.arange(L, device=_device)

    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_loss = -logprobs[idx, shift_labels]     # (T-1,) кросс-энтропия на токен

    # ранг истинного токена = число токенов словаря строго вероятнее него
    true_logit = shift_logits[idx, shift_labels].unsqueeze(1)   # (T-1, 1)
    ranks = (shift_logits > true_logit).sum(dim=1)              # (T-1,) 0-based

    losses = token_loss.cpu().numpy()
    ranks_np = ranks.cpu().numpy()

    # --- perplexity ---
    perplexity = float(np.exp(losses.mean()))

    # --- rank_entropy: энтропия долей по корзинам рангов ---
    # bins: [0]=top-1, [1]=2..10, [2]=11..100, [3]=101..1000, [4]=>1000
    buckets = np.digitize(ranks_np, RANK_BINS)    # 0..len(RANK_BINS)
    counts = np.bincount(buckets, minlength=len(RANK_BINS) + 1).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    rank_entropy = float(-(nz * np.log2(nz)).sum())   # биты, max ~ log2(5)=2.32

    # --- perplexity_std: перплексия по предложениям -> std ---
    # offsets[i+1] — символьная позиция предсказанного токена (label = токен i+1)
    sents = [(s.start, s.stop) for s in sentenize(text)]
    perp_by_sent = {}
    if len(sents) >= 2:
        starts = [offsets[i + 1][0] for i in range(L)]
        sent_loss = {}
        for i, ch in enumerate(starts):
            # к какому предложению относится символ ch
            si = None
            for j, (a, b) in enumerate(sents):
                if a <= ch < b:
                    si = j
                    break
            if si is None:
                continue
            sent_loss.setdefault(si, []).append(losses[i])
        perp_by_sent = {
            si: math.exp(sum(ls) / len(ls)) for si, ls in sent_loss.items() if ls
        }
    if len(perp_by_sent) >= 2:
        perplexity_std = float(np.std(list(perp_by_sent.values())))
    else:
        perplexity_std = 0.0

    return perplexity, perplexity_std, rank_entropy


def _words(text: str) -> list[str]:
    return [t.text.lower() for t in tokenize(text) if t.text.isalpha()]


def burstiness_and_lengths(text: str) -> tuple[float, float]:
    """std и mean длин предложений (в словах). Люди пишут рвано -> высокий std."""
    sents = list(sentenize(text))
    lengths = [len(_words(s.text)) for s in sents]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 2:
        return 0.0, float(lengths[0]) if lengths else 0.0
    mean = sum(lengths) / len(lengths)
    var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(var), mean


def mattr(words: list[str], window: int = 20) -> float:
    """
    Moving-Average TTR — лексическое разнообразие, НЕ зависящее от длины текста.

    Почему не обычный TTR (уник/всего): сырой TTR механически ПАДАЕТ с длиной
    (чем длиннее текст, тем чаще повторы), поэтому на коротких текстах он раздут
    и работает как «прокси длины», а не авторства — ровно тот артефакт, который
    тащил 0.5 важности в прошлой модели.

    MATTR убирает это: считаем TTR в скользящем окне ФИКСИРОВАННОЙ длины и
    усредняем. Каждое измерение на одинаковом числе токенов -> длина текста
    больше не искажает значение.

    Выбор окна=20: из MTLD / MATTR / TTR-на-окне взял MATTR, т.к. он (а) полностью
    снимает длинно-артефакт (фикс. окно), (б) детерминирован и прост, (в) в отличие
    от MTLD устойчив на КОРОТКИХ текстах — а у нас медиана ~20 слов и слабое место
    именно короткие тексты, где MTLD (рассчитан на >=50 токенов) шумит. Окно 20
    подобрано под эту медиану, чтобы под честный MATTR попадало большинство текстов.
    Тексты короче окна — честно считаем обычный TTR (редкий случай, мало токенов).
    """
    n = len(words)
    if n == 0:
        return 0.0
    if n < window:
        return len(set(words)) / n
    ratios = [len(set(words[i:i + window])) / window for i in range(n - window + 1)]
    return sum(ratios) / len(ratios)


def cliche_rate(text: str, n_words: int) -> float:
    low = text.lower()
    hits = sum(low.count(c) for c in CLICHES)
    return (hits / n_words * 100) if n_words else 0.0


def punctuation_variety(text: str) -> float:
    """Сколько разных знаков препинания используется (норм. на 8 базовых)."""
    found = set(re.findall(r"[.,;:!?—()\"]", text))
    return len(found) / 8.0


def extract(text: str) -> Features:
    words = _words(text)
    n = len(words)
    burst, mean_len = burstiness_and_lengths(text)
    lm = _lm_token_stats(text)
    if lm is None:
        perplexity, perplexity_std, rank_entropy = 0.0, 0.0, 0.0
    else:
        perplexity, perplexity_std, rank_entropy = lm
    return Features(
        perplexity=perplexity,
        perplexity_std=perplexity_std,
        rank_entropy=rank_entropy,
        burstiness=burst,
        mean_sent_len=mean_len,
        mattr=mattr(words),
        cliche_rate=cliche_rate(text, n),
        punct_variety=punctuation_variety(text),
    )


def to_vector(f: Features) -> list[float]:
    """Фиксированный порядок фич — важно, чтобы обучение и инференс совпадали."""
    return [
        f.perplexity, f.perplexity_std, f.rank_entropy,
        f.burstiness, f.mean_sent_len, f.mattr,
        f.cliche_rate, f.punct_variety,
    ]


FEATURE_NAMES = list(asdict(Features(0, 0, 0, 0, 0, 0, 0, 0)).keys())
