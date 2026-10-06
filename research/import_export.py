"""Export/restore research/db/research.sqlite <-> research/export/*.jsonl.gz (abstracts are not exported).
usage: python research/import_export.py export|import   (default: import)"""
import gzip, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(__file__))
import harvest
D = os.path.join(os.path.dirname(__file__), "export")
TABLES = ("papers", "repos", "hits", "queries_done", "reads")
mode = sys.argv[1] if len(sys.argv) > 1 else "import"
c = harvest.db()
c.execute("CREATE TABLE IF NOT EXISTS reads(ref TEXT PRIMARY KEY, kind TEXT, level TEXT, url TEXT, title TEXT, hypotheses TEXT, stance TEXT, note TEXT, session INT)")
if mode == "export":
    for t in TABLES:
        cols = [r[1] for r in c.execute(f"pragma table_info({t})") if r[1] != "abstract"]
        n = 0
        with gzip.open(os.path.join(D, f"{t}.jsonl.gz"), "wt") as f:
            for row in c.execute(f"SELECT {','.join(cols)} FROM {t}"):
                f.write(json.dumps(dict(zip(cols, row)), ensure_ascii=False) + "\n"); n += 1
        print("exported", t, n)
else:
    for t in TABLES:
        for line in gzip.open(os.path.join(D, f"{t}.jsonl.gz"), "rt"):
            r = json.loads(line)
            c.execute(f"INSERT OR IGNORE INTO {t}({','.join(r)}) VALUES({','.join('?'*len(r))})", list(r.values()))
    c.commit(); print("restored")
