"""
Hypothesis-driven literature & repository harvester.

Honesty contract (principle #28):
  * `papers` / `repos` rows = metadata+abstract HARVESTED (triage level L0).
  * `triage` column: L0 harvested, L1 auto-scored by relevance heuristic over abstract,
    L2 = abstract/README read & annotated by the researcher (notes in research/notes),
    L3 = full paper / code inspected.
  * Reports must distinguish these levels. Harvested != read.

Sources: arXiv API, OpenAlex API, GitHub search API (star-stratified, principle #4).
Every row records which (problem, hypothesis, kind, query) found it -> provenance.
Re-running is idempotent (dedup by arXiv id / DOI / normalized title / repo full_name).
"""
import json, os, re, sqlite3, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import TAXONOMY, GITHUB_QUERIES, STAR_BUCKETS

DB = os.path.join(os.path.dirname(__file__), "db", "research.sqlite")
UA = {"User-Agent": "natsu-uta-research/0.1 (mailto:research@example.org)"}


def gh_token():
    p = os.path.expanduser("~/.git-credentials")
    if os.path.exists(p):
        m = re.search(r"://[^:]+:([^@]+)@github.com", open(p).read())
        if m:
            return m.group(1)
    return os.environ.get("GITHUB_TOKEN")


def db():
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS papers(
      key TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INT, venue TEXT, url TEXT,
      source TEXT, cited_by INT, triage TEXT DEFAULT 'L0', relevance REAL);
    CREATE TABLE IF NOT EXISTS repos(
      full_name TEXT PRIMARY KEY, description TEXT, stars INT, forks INT, language TEXT,
      created TEXT, pushed TEXT, url TEXT, archived INT, topics TEXT, triage TEXT DEFAULT 'L0', relevance REAL);
    CREATE TABLE IF NOT EXISTS hits(
      item TEXT, item_type TEXT, problem TEXT, hypothesis TEXT, kind TEXT, query TEXT, source TEXT,
      PRIMARY KEY(item, item_type, query, source));
    CREATE TABLE IF NOT EXISTS queries_done(query TEXT, source TEXT, n INT, ts REAL, PRIMARY KEY(query, source));
    """)
    return c


def get(url, headers=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # rate limit / transient
            wait = 10 * (i + 1)
            if hasattr(e, "code") and e.code in (403, 429):
                wait = 65
            print(f"  ! {e} -> retry in {wait}s", flush=True)
            time.sleep(wait)
    return None


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]


def done(c, q, src):
    return c.execute("SELECT 1 FROM queries_done WHERE query=? AND source=?", (q, src)).fetchone() is not None


# ----------------------------------------------------------------- arXiv
ARXIV_FAILS = [0]


def arxiv(c, q, prov, n=100):
    if done(c, q, "arxiv") or ARXIV_FAILS[0] >= 3:   # arXiv API throttles hard; back off for this run
        return 0
    terms = "+AND+".join(f"all:{urllib.parse.quote(w)}" for w in q.split() if len(w) > 2)
    url = f"https://export.arxiv.org/api/query?search_query={terms}&start=0&max_results={n}&sortBy=relevance"
    raw = get(url, tries=2)
    time.sleep(3.1)
    if raw is None:
        ARXIV_FAILS[0] += 1
        return 0
    ARXIV_FAILS[0] = 0
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return 0
    k = 0
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1].split("v")[0]
        title = " ".join((e.find("a:title", ns).text or "").split())
        abs_ = " ".join((e.find("a:summary", ns).text or "").split())
        year = int(e.find("a:published", ns).text[:4])
        key = "t:" + norm_title(title)
        c.execute("INSERT OR IGNORE INTO papers(key,title,abstract,year,venue,url,source) VALUES(?,?,?,?,?,?,?)",
                  (key, title, abs_, year, "arXiv", f"https://arxiv.org/abs/{aid}", "arxiv"))
        c.execute("INSERT OR IGNORE INTO hits VALUES(?,?,?,?,?,?,?)", (key, "paper", *prov, q, "arxiv"))
        k += 1
    c.execute("INSERT OR REPLACE INTO queries_done VALUES(?,?,?,?)", (q, "arxiv", k, time.time()))
    c.commit()
    return k


# ----------------------------------------------------------------- OpenAlex
def inv_abs(ix):
    if not ix:
        return ""
    pos = {}
    for w, ps in ix.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


OA_DEAD = [False]


def openalex(c, q, prov, n=100):
    if done(c, q, "openalex") or OA_DEAD[0]:
        return 0
    url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": q, "per-page": n, "filter": "from_publication_date:2015-01-01",
        "mailto": "research@example.org"}))
    raw = get(url, tries=1)
    if raw is None:
        OA_DEAD[0] = True   # daily budget exhausted -> skip for the rest of this run (logged, retried next run)
        return 0
    k = 0
    for w in json.loads(raw).get("results", []):
        title = w.get("display_name") or ""
        if not title:
            continue
        key = "t:" + norm_title(title)
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        c.execute("INSERT OR IGNORE INTO papers(key,title,abstract,year,venue,url,source,cited_by) VALUES(?,?,?,?,?,?,?,?)",
                  (key, title, inv_abs(w.get("abstract_inverted_index")), w.get("publication_year"), venue,
                   w.get("doi") or w.get("id"), "openalex", w.get("cited_by_count")))
        c.execute("UPDATE papers SET cited_by=COALESCE(cited_by, ?) WHERE key=?", (w.get("cited_by_count"), key))
        c.execute("INSERT OR IGNORE INTO hits VALUES(?,?,?,?,?,?,?)", (key, "paper", *prov, q, "openalex"))
        k += 1
    c.execute("INSERT OR REPLACE INTO queries_done VALUES(?,?,?,?)", (q, "openalex", k, time.time()))
    c.commit()
    time.sleep(0.3)
    return k


# ----------------------------------------------------------------- Crossref (fallback; includes venues/DOIs)
def crossref(c, q, prov, n=100):
    if done(c, q, "crossref"):
        return 0
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": n, "filter": "from-pub-date:2015-01-01",
         "select": "DOI,title,abstract,issued,container-title,is-referenced-by-count", "mailto": "research@example.org"})
    raw = get(url, tries=2)
    if raw is None:
        return 0
    k = 0
    for w in json.loads(raw).get("message", {}).get("items", []):
        title = " ".join((w.get("title") or [""])[0].split())
        if not title:
            continue
        key = "t:" + norm_title(title)
        yr = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        abs_ = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
        c.execute("INSERT OR IGNORE INTO papers(key,title,abstract,year,venue,url,source,cited_by) VALUES(?,?,?,?,?,?,?,?)",
                  (key, title, " ".join(abs_.split()), yr, (w.get("container-title") or [None])[0],
                   "https://doi.org/" + w["DOI"], "crossref", w.get("is-referenced-by-count")))
        c.execute("INSERT OR IGNORE INTO hits VALUES(?,?,?,?,?,?,?)", (key, "paper", *prov, q, "crossref"))
        k += 1
    c.execute("INSERT OR REPLACE INTO queries_done VALUES(?,?,?,?)", (q, "crossref", k, time.time()))
    c.commit()
    time.sleep(0.5)
    return k


# ----------------------------------------------------------------- GitHub
def github(c, q, bucket, tok):
    qq = f"{q} {bucket}"
    if done(c, qq, "github"):
        return 0
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": qq, "per_page": 100, "sort": "updated"})
    raw = get(url, {"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"} if tok else None)
    time.sleep(2.2)  # 30 req/min authenticated
    if raw is None:
        return 0
    k = 0
    for r in json.loads(raw).get("items", []):
        c.execute("INSERT OR IGNORE INTO repos(full_name,description,stars,forks,language,created,pushed,url,archived,topics)"
                  " VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (r["full_name"], r.get("description"), r["stargazers_count"], r["forks_count"], r.get("language"),
                   r["created_at"], r["pushed_at"], r["html_url"], int(r.get("archived", False)), ",".join(r.get("topics", []))))
        c.execute("INSERT OR IGNORE INTO hits VALUES(?,?,?,?,?,?,?)", (r["full_name"], "repo", "gh", q, bucket, qq, "github"))
        k += 1
    c.execute("INSERT OR REPLACE INTO queries_done VALUES(?,?,?,?)", (qq, "github", k, time.time()))
    c.commit()
    return k


def main(which):
    c = db()
    if which in ("papers", "all"):
        for P, pd in TAXONOMY.items():
            for H, hd in pd["hypotheses"].items():
                for kind in ("support", "refute"):
                    for q in hd.get(kind, []):
                        a = arxiv(c, q, (P, H, kind))
                        o = openalex(c, q, (P, H, kind))
                        x = crossref(c, q, (P, H, kind))
                        print(f"[paper] {H} {kind:7s} '{q}': arxiv={a} openalex={o} crossref={x}", flush=True)
    if which in ("repos", "all"):
        tok = gh_token()
        for q in GITHUB_QUERIES:
            for b in STAR_BUCKETS:
                k = github(c, q, b, tok)
                print(f"[repo] '{q}' {b}: {k}", flush=True)
    print("TOTAL papers", c.execute("select count(*) from papers").fetchone()[0],
          "repos", c.execute("select count(*) from repos").fetchone()[0])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
