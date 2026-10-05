"""Rebuild research/db/research.sqlite from research/export/*.jsonl.gz (abstracts are not exported)."""
import gzip, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(__file__))
import harvest
D = os.path.join(os.path.dirname(__file__), "export")
c = harvest.db()
c.execute("CREATE TABLE IF NOT EXISTS reads(ref TEXT PRIMARY KEY, kind TEXT, level TEXT, url TEXT, title TEXT, hypotheses TEXT, stance TEXT, note TEXT, session INT)")
for t in ("papers", "repos", "hits", "queries_done", "reads"):
    for line in gzip.open(os.path.join(D, f"{t}.jsonl.gz"), "rt"):
        r = json.loads(line)
        c.execute(f"INSERT OR IGNORE INTO {t}({','.join(r)}) VALUES({','.join('?'*len(r))})", list(r.values()))
c.commit(); print("restored")
