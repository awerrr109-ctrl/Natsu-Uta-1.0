"""
L1 triage over the harvested corpus + knowledge-map statistics.

L1 = automatic relevance score from title/abstract (or README description) keyword
evidence per hypothesis. It is NOT reading. It is used to (a) prune off-topic hits
(OpenAlex full-text search is noisy), (b) prioritise the L2 reading queue,
(c) surface low-star repos that match rare hypotheses (principle #4).

Outputs:
  research/notes/corpus_stats.md     counts per source / hypothesis / kind / star bucket
  research/notes/l2_queue.md         top-N per hypothesis for human/agent reading (support AND refute)
  research/notes/lowstar_gems.md     relevant repos with <=10 stars
"""
import os, re, sqlite3, sys, collections, math
sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import TAXONOMY

DB = os.path.join(os.path.dirname(__file__), "db", "research.sqlite")
NOTES = os.path.join(os.path.dirname(__file__), "notes")
CORE = re.compile(r"\b(language model|llm|transformer|attention|token|pretrain|large language|neural network|recurrent|state space|mixture of experts|decoding|reasoning)\b", re.I)
ML_CTX = re.compile(r"(llm|language model|transformer|neural|deep learning|pytorch|torch|jax|attention|gpt|bert|"
                    r"machine learning|\\bml\\b|\\bai\\b|model|training|inference|cuda|triton|token|nlp|diffusion|reinforcement)", re.I)
LM_CTX = re.compile(r"(language model|\bllms?\b|\bnlp\b|natural language|text generation|next[- ]token|pretraining corpus|"
                    r"perplexity|in-context|chain[- ]of[- ]thought|\bgpt|instruction[- ]tun|question answering|machine translation)", re.I)
OFF_DOMAIN = re.compile(r"(image|vision|visual|video|pixel|segmentation|detection|speech|audio|spectral|medical|clinical|"
                        r"remote sensing|hyperspectral|point cloud|molecul|protein|traffic|fault diagnosis|power system|"
                        r"electrical transformer|aero-engine|super-resolution|restoration)", re.I)
STOP = set("the a an of for and in on to with via by from is are be llm language model models large".split())


def terms(q):
    return [w for w in re.findall(r"[a-z0-9\-]+", q.lower()) if w not in STOP and len(w) > 2]


def score(text, q):
    t = (text or "").lower()
    ts = terms(q)
    if not ts:
        return 0.0
    hit = sum(1 for w in ts if w in t)
    base = hit / len(ts)
    # L1 v2 (session 2): "transformer" alone matched vision/speech/engineering papers at rel=1.0 and flooded the L2 queue.
    # Domain-aware context factor: LM evidence 1.0; generic neural context 0.6; off-domain without LM evidence 0.2.
    if LM_CTX.search(t):
        ctx = 1.0
    elif OFF_DOMAIN.search(t):
        ctx = 0.2
    elif CORE.search(t):
        ctx = 0.6
    else:
        ctx = 0.3
    return base * ctx


def main():
    c = sqlite3.connect(DB)
    os.makedirs(NOTES, exist_ok=True)
    # ---- L1 scoring for papers
    rows = c.execute("SELECT p.key, p.title, p.abstract, h.query FROM papers p JOIN hits h ON h.item=p.key AND h.item_type='paper'").fetchall()
    best = collections.defaultdict(float)
    for k, t, a, q in rows:
        best[k] = max(best[k], score((t or "") + " " + (a or ""), q))
    c.executemany("UPDATE papers SET relevance=?, triage=CASE WHEN triage='L0' THEN 'L1' ELSE triage END WHERE key=?",
                  [(v, k) for k, v in best.items()])
    rows = c.execute("SELECT r.full_name, r.description, r.topics, h.hypothesis FROM repos r JOIN hits h ON h.item=r.full_name AND h.item_type='repo'").fetchall()
    rb = collections.defaultdict(float)
    for k, d, tp, q in rows:
        txt = (k.replace("/", " ").replace("-", " ") + " " + (d or "") + " " + (tp or "").replace(",", " "))
        s = sum(1 for w in terms(q) if w in txt.lower()) / max(1, len(terms(q)))
        # require ML context; otherwise e.g. 'mamba' matches web frameworks / snakes
        if not ML_CTX.search(txt):
            s *= 0.3
        rb[k] = max(rb[k], s)
    c.executemany("UPDATE repos SET relevance=?, triage=CASE WHEN triage='L0' THEN 'L1' ELSE triage END WHERE full_name=?",
                  [(v, k) for k, v in rb.items()])
    c.commit()

    # ---- stats
    L = []
    np_ = c.execute("select count(*) from papers").fetchone()[0]
    nr = c.execute("select count(*) from repos").fetchone()[0]
    L.append(f"# Corpus statistics (auto-generated)\n\nHonesty note: harvested(L0) != read. Levels: L0 harvested, L1 auto-scored, L2 abstract/README read+annotated, L3 full text/code inspected.\n")
    L.append(f"- unique papers harvested: **{np_}**")
    L.append(f"- unique repos harvested: **{nr}**")
    for lvl in ("L0", "L1", "L2", "L3"):
        L.append(f"- papers at {lvl}: {c.execute('select count(*) from papers where triage=?', (lvl,)).fetchone()[0]}; repos at {lvl}: {c.execute('select count(*) from repos where triage=?', (lvl,)).fetchone()[0]}")
    rel_p = c.execute("select count(*) from papers where relevance>=0.6").fetchone()[0]
    rel_r = c.execute("select count(*) from repos where relevance>=0.5").fetchone()[0]
    L.append(f"- papers with L1 relevance>=0.6 (on-topic estimate): {rel_p}")
    L.append(f"- repos with L1 relevance>=0.5: {rel_r}")
    L.append("\n## Queries executed\n")
    for s, n, tot in c.execute("select source, count(*), sum(n) from queries_done group by source"):
        L.append(f"- {s}: {n} queries, {tot} raw results")
    L.append("\n## Papers by source\n")
    for s, n in c.execute("select source,count(*) from papers group by source"):
        L.append(f"- {s}: {n}")
    L.append("\n## Papers per hypothesis × kind (unique)\n\n| hypothesis | support | refute |\n|---|---|---|")
    for H, in c.execute("select distinct hypothesis from hits where item_type='paper' order by hypothesis"):
        s = c.execute("select count(distinct item) from hits where hypothesis=? and kind='support'", (H,)).fetchone()[0]
        r = c.execute("select count(distinct item) from hits where hypothesis=? and kind='refute'", (H,)).fetchone()[0]
        L.append(f"| {H} | {s} | {r} |")
    L.append("\n## Repos by star bucket\n\n| bucket | repos |\n|---|---|")
    for lo, hi in [(0, 1), (2, 10), (11, 100), (101, 1000), (1001, 10 ** 9)]:
        L.append(f"| {lo}-{hi if hi < 10**9 else '∞'} | {c.execute('select count(*) from repos where stars between ? and ?', (lo, hi)).fetchone()[0]} |")
    L.append(f"\n- archived repos: {c.execute('select count(*) from repos where archived=1').fetchone()[0]}")
    new_r = c.execute("select count(*) from repos where created>='2026-01-01'").fetchone()[0]
    old_r = c.execute("select count(*) from repos where pushed<'2024-01-01'").fetchone()[0]
    L.append(f"- repos created >= 2026-01-01: {new_r}")
    L.append(f"- repos last pushed < 2024-01-01 (old/abandoned): {old_r}")
    L.append("\n## Year histogram (papers)\n")
    for y, n in c.execute("select year,count(*) from papers where year>=2015 group by year order by year"):
        L.append(f"- {y}: {n}")
    open(os.path.join(NOTES, "corpus_stats.md"), "w").write("\n".join(L) + "\n")

    # ---- L2 reading queue
    Q = ["# L2 reading queue (top L1-relevance per hypothesis; support and refute)\n"]
    for P, pd in TAXONOMY.items():
        for H in pd["hypotheses"]:
            Q.append(f"\n## {H}\n")
            for kind in ("support", "refute"):
                Q.append(f"\n### {kind}\n")
                for t, u, y, rel, cb in c.execute(
                        "select distinct p.title,p.url,p.year,p.relevance,p.cited_by from papers p join hits h on h.item=p.key "
                        "where h.hypothesis=? and h.kind=? order by p.relevance desc, coalesce(p.cited_by,0) desc limit 12", (H, kind)):
                    Q.append(f"- [{y}] {t} — {u} (rel={rel:.2f}, cites={cb})")
    open(os.path.join(NOTES, "l2_queue.md"), "w").write("\n".join(Q) + "\n")

    G = ["# Low-star relevant repos (<=10 stars, relevance>=0.6) — principle #4\n"]
    for fn, d, s, pu, rel in c.execute("select full_name,description,stars,pushed,relevance from repos where stars<=10 and relevance>=0.6 order by pushed desc limit 400"):
        G.append(f"- {fn} ★{s} pushed {pu[:10]} — {(d or '')[:140]}")
    open(os.path.join(NOTES, "lowstar_gems.md"), "w").write("\n".join(G) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
