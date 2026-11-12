"""Check references.bib against Crossref (DOI entries) and arXiv (eprint entries).

Prints one line per entry: OK, or the fields that disagree with the registry.
Entries with neither a DOI nor an eprint are looked up by title on Crossref and
the best match is printed for manual inspection.
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "references.bib"
UA = {"User-Agent": "lit-review-bib-check/1.0 (mailto:noreply@example.com)"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.groups()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append((typ.lower(), key.strip(), fields))
    return entries


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+|[{}$\\'\"`^~]", "", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise err


def check_doi(key, f):
    try:
        d = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(f["doi"])))["message"]
    except Exception as e:  # noqa: BLE001
        return f"{key}: DOI LOOKUP FAILED ({e})"
    issues = []
    title = (d.get("title") or [""])[0]
    if sim(title, f.get("title", "")) < 0.85:
        issues.append(f"title: bib='{f.get('title')}' | crossref='{title}'")
    parts = (d.get("published-print") or d.get("published-online") or d.get("issued") or {}).get("date-parts", [[None]])
    years = {str(x.get("date-parts", [[None]])[0][0]) for x in
             (d.get("published-print", {}), d.get("published-online", {}), d.get("issued", {})) if x}
    if f.get("year") and f["year"] not in years:
        issues.append(f"year: bib={f['year']} | crossref={sorted(years)}")
    if f.get("volume") and d.get("volume") and norm(f["volume"]) != norm(d["volume"]):
        issues.append(f"volume: bib={f['volume']} | crossref={d['volume']}")
    cr_pages = d.get("page") or d.get("article-number")
    if f.get("pages") and cr_pages:
        bp = norm(f["pages"].split("--")[0]).replace(" ", "")
        cp = norm(cr_pages.split("-")[0]).replace(" ", "")
        an = norm(d.get("article-number", "")).replace(" ", "")
        if bp not in (cp, an) and not cp.endswith(bp) and not bp.endswith(cp):
            issues.append(f"pages: bib={f['pages']} | crossref={cr_pages} / art={d.get('article-number')}")
    if f.get("number") and d.get("issue") and norm(f["number"]) != norm(d["issue"]):
        issues.append(f"number: bib={f['number']} | crossref={d['issue']}")
    fa = norm(f.get("author", "").split(" and ")[0].split(",")[0])
    ca = norm((d.get("author") or [{}])[0].get("family", ""))
    if fa and ca and fa != ca and sim(fa, ca) < 0.8:
        issues.append(f"first author: bib={fa} | crossref={ca}")
    return f"{key}: OK" if not issues else f"{key}: " + "; ".join(issues)


def check_arxiv(key, f):
    try:
        time.sleep(3.2)
        x = get("https://export.arxiv.org/api/query?id_list=" + f["eprint"])
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(x).find("a:entry", ns)
        title = " ".join(e.find("a:title", ns).text.split())
        first = e.find("a:author/a:name", ns).text
    except Exception as ex:  # noqa: BLE001
        return f"{key}: ARXIV LOOKUP FAILED ({ex})"
    issues = []
    if sim(title, f.get("title", "")) < 0.85:
        issues.append(f"title: bib='{f.get('title')}' | arxiv='{title}'")
    fa = norm(f.get("author", "").split(" and ")[0].split(",")[0])
    if fa and fa not in norm(first):
        issues.append(f"first author: bib={fa} | arxiv={first}")
    return f"{key}: OK (arXiv)" if not issues else f"{key}: " + "; ".join(issues)


def check_search(key, f):
    q = urllib.parse.quote(f.get("title", ""))
    try:
        items = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"))["message"]["items"]
    except Exception as e:  # noqa: BLE001
        return f"{key}: SEARCH FAILED ({e})"
    best = max(items, key=lambda it: sim((it.get("title") or [""])[0], f.get("title", "")), default=None)
    if best is None:
        return f"{key}: NO MATCH"
    t = (best.get("title") or [""])[0]
    s = sim(t, f.get("title", ""))
    tag = "FOUND" if s > 0.9 else "CHECK"
    return f"{key}: {tag} (no DOI) sim={s:.2f} -> '{t}' [{best.get('DOI')}]"


def check(entry):
    typ, key, f = entry
    if "doi" in f:
        return check_doi(key, f)
    if "eprint" in f:
        return check_arxiv(key, f)
    return check_search(key, f)


if __name__ == "__main__":
    entries = parse_bib(BIB.read_text())
    only = set(sys.argv[1:])
    if only:
        entries = [e for e in entries if e[1] in only]
    print(f"{len(entries)} entries")
    with ThreadPoolExecutor(6) as ex:
        for line in ex.map(check, entries):
            print(line, flush=True)
