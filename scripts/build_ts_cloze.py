"""NOTE: after building, items whose context prefix (80 chars) occurs in ts_train.txt are removed (13% contaminated; TinyStories openings repeat).
Build a held-out TinyStories cloze MC set (downstream check beyond bpb; R31/R40 motivated).
Item: context = first k sentences of a validation story; choices = the true next sentence + 3 sentences from other stories (same length band).
Uses only the validation split (never trained on). Also emits a 'name-consistency' subset where the true continuation reuses a character
name from the context and distractors use a different name, which needs in-context retrieval (cf. E6/H14.1)."""
import json, random, re, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
txt = open(os.path.join(ROOT, "data_cache/ts_valid.txt"), encoding="utf-8", errors="ignore").read()
stories = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200]
rng = random.Random(2026)
split = lambda s: [x.strip() for x in re.split(r"(?<=[.!?])\s+", s.replace("\n", " ")) if 20 <= len(x.strip()) <= 140]
pool = [x for s in stories[5000:9000] for x in split(s)]
items, names = [], []
NAME = re.compile(r"\b(Tom|Lily|Sam|Ben|Mia|Tim|Sue|Anna|Max|Lucy|Jack|Sara|Amy|Bob|Kate|Mom|Dad)\b")
for s in stories[:5000]:
    ss = split(s)
    if len(ss) < 5:
        continue
    k = rng.randint(2, min(5, len(ss) - 2)); ctx, gold = " ".join(ss[:k]), ss[k]
    L = len(gold); cand = [x for x in rng.sample(pool, 60) if abs(len(x) - L) < 25 and x != gold][:3]
    if len(cand) < 3:
        continue
    ch = cand + [gold]; rng.shuffle(ch)
    items.append({"ctx": ctx + " ", "choices": ch, "label": ch.index(gold)})
    m_ctx, m_gold = set(NAME.findall(ctx)), NAME.findall(gold)
    if m_gold and m_gold[0] in m_ctx and m_gold[0] not in ("Mom", "Dad"):
        others = [n for n in ["Tom", "Lily", "Sam", "Ben", "Mia", "Tim", "Sue", "Anna", "Max", "Lucy"] if n not in m_ctx]
        alts = [re.sub(r"\b%s\b" % m_gold[0], o, gold) for o in rng.sample(others, 3)]
        ch2 = alts + [gold]; rng.shuffle(ch2)
        names.append({"ctx": ctx + " ", "choices": ch2, "label": ch2.index(gold)})
    if len(items) >= 400 and len(names) >= 200:
        break
os.makedirs(os.path.join(ROOT, "data_cache/evals"), exist_ok=True)
json.dump(items[:400], open(os.path.join(ROOT, "data_cache/evals/ts_cloze.json"), "w"))
json.dump(names[:200], open(os.path.join(ROOT, "data_cache/evals/ts_names.json"), "w"))
print("cloze", len(items[:400]), "names", len(names[:200]))
print(json.dumps(names[0], ensure_ascii=False)[:400])
