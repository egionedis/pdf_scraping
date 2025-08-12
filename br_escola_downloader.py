# br_escola_downloader.py
# Downloader for https://vestibular.brasilescola.uol.com.br/downloads
# - Region page -> per-university page -> per-link dump
# - ALSO supports extra downloads pages via --pages (treated as top-level "regions", e.g., OBMEP)
# - Windows-safe folder names, resumable downloads, nested extraction with 7-Zip
# - Long-path safe on Windows (\\?\ prefix)
# - Progress logs + total elapsed timer

import os, re, pathlib, argparse, shutil, subprocess, zipfile, rarfile, threading, uuid, time
from urllib.parse import urljoin, urlparse, unquote
import http.client as httplib
from time import sleep

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# Silence TLS warnings (we disable cert verification below)
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

BASE     = "https://vestibular.brasilescola.uol.com.br"
START    = f"{BASE}/downloads"
OUT_ROOT = pathlib.Path("downloads")
SEVENZ_PATH = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BrasilEscolaDownloader/7.8)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Site region slugs (note: "centrooeste" has no hyphen on the site)
REGION_SLUGS = {"centrooeste","nordeste","norte","sudeste","sul"}

# Optional friendly renames for long page slugs
FRIENDLY_PAGE_SLUG = {
    "olimpiada-brasileira-matematica-escolas-publicas": "obmep",
}

WORK_BASE = pathlib.Path("w")
WORK_BASE.mkdir(exist_ok=True)

_print_lock = threading.Lock()
def _print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

def _short_workdir() -> pathlib.Path:
    d = WORK_BASE / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------- Windows long-path helpers ----------
def _win_long(p: pathlib.Path) -> str:
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s

def exists_long(p: pathlib.Path) -> bool:
    try:
        return os.path.exists(_win_long(p))
    except Exception:
        return False

# ---------- HTTP ----------
def make_session():
    s = requests.Session()
    retries = Retry(total=5, connect=5, read=5, backoff_factor=0.8,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods=["GET","HEAD"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    s.verify = False
    return s

def soup_get(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

# ---------- discovery ----------
def collect_region_pages(session):
    """Return [(slug, url)] for region pages present on START."""
    soup = soup_get(session, START)
    out = []
    for a in soup.select('a[href*="/downloads/"][href$=".htm"]'):
        url = urljoin(BASE, a.get("href",""))
        slug = os.path.splitext(os.path.basename(urlparse(url).path))[0]
        if slug in REGION_SLUGS:
            out.append((slug, url))
    seen=set(); uniq=[]
    for n,u in out:
        if u not in seen:
            uniq.append((n,u)); seen.add(u)
    return uniq

def derive_alias_from_h4(h4_text: str, uni_name: str) -> str:
    t = (h4_text or "").strip()
    m = re.search(r"Vestibulares?\s+(?:do|da|de|dos|das)\s+(.+?)(?:[.;]|$)", t, flags=re.I)
    if m: return m.group(1).strip()
    m = re.search(r"Vestibulares?\s+([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 \-]+?)(?:[.;]|$)", t, flags=re.I)
    if m: return m.group(1).strip()
    return uni_name.strip()

def collect_universities_in_region(session, region_url):
    """On a region page, collect links to per-university downloads pages -> [(display, url, alias)]."""
    soup = soup_get(session, region_url)
    out = []
    for a in soup.select('a[href*="/downloads/"][href$=".htm"]'):
        href = a.get("href","") or ""
        url  = urljoin(BASE, href)
        slug = os.path.splitext(os.path.basename(urlparse(url).path))[0]
        if slug in REGION_SLUGS:
            continue
        text = a.get_text(" ", strip=True) or ""
        parts = re.split(r'\s+Provas?\b', text, maxsplit=1, flags=re.I)
        uni_display = parts[0].strip() if parts and parts[0].strip() else slug.replace("-", " ")
        alias = derive_alias_from_h4(text, uni_display)
        out.append((uni_display, url, alias))
    seen=set(); uniq=[]
    for rec in out:
        if rec[1] not in seen:
            uniq.append(rec); seen.add(rec[1])
    return uniq

# ---------- label cleanup / safe paths ----------
DL_RX = re.compile(
    r"""(?ix)
    (?:^|\s)\d{1,3}(?:[.\s]\d{3})*\s+downloads?\s+realizad[oa]s\.?
    """)

def _clean_label(label: str, fallback: str) -> str:
    s = (label or "").strip()
    if not s:
        s = fallback
    s = DL_RX.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -–—•\t\r\n")
    if not re.search(r"[A-Za-zÀ-ÿ]", s):  # avoid folders named only "47483"
        s = fallback
    return s or fallback or "arquivo"

INVALID_CHARS_RX = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
WIN_RESERVED = {
    "CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
    "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"
}
def safe_segment(name: str, fallback: str = "arquivo", max_len: int = 120) -> str:
    s = _clean_label(name, fallback)
    s = INVALID_CHARS_RX.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" .")
    if not s:
        s = fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    if s.upper() in WIN_RESERVED:
        s = "_" + s
    return s

# ---------- item collection ----------
def collect_items(session, page_url):
    """From any downloads page, collect actual downloadable items -> [(label, url)]."""
    soup = soup_get(session, page_url)

    def is_download_href(h):
        h = (h or "").strip().lower()
        return ("/baixar/" in h) or h.endswith((".pdf",".zip",".rar"))

    anchors = [a for a in soup.select("a[href]") if is_download_href(a.get("href"))]

    # Map URL -> texts; pick the longest per-URL
    url_to_texts = {}
    for a in anchors:
        href = urljoin(BASE, a.get("href",""))
        texts = url_to_texts.setdefault(href, [])
        t = a.get_text(" ", strip=True)
        if t: texts.append(t)
        if a.get("title"): texts.append(a["title"])
        img = a.find("img")
        if img and img.get("alt"): texts.append(img["alt"])

    items = []
    for a in anchors:
        url  = urljoin(BASE, a.get("href",""))
        candidates = url_to_texts.get(url, [])
        fallback = os.path.basename(urlparse(url).path) or "arquivo"
        raw = max(candidates, key=len) if candidates else fallback
        label = _clean_label(raw, fallback)
        items.append((label, url))

    # de-dupe by URL
    seen=set(); uniq=[]
    for label, url in items:
        if url not in seen:
            uniq.append((label, url)); seen.add(url)
    return uniq

# ---------- filename ext sniff ----------
def _sniff_name_and_ext(session, file_url, headers):
    try:
        with session.get(file_url, headers=headers, stream=True, timeout=30) as r0:
            r0.raise_for_status()
            cd = r0.headers.get("Content-Disposition","")
            m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
            resp_name = unquote(m.group(1)) if m else os.path.basename(urlparse(r0.url or file_url).path)
            ext = os.path.splitext(resp_name)[1].lower()
            if not ext:
                ct = (r0.headers.get("Content-Type","") or "").lower()
                if "zip" in ct: ext = ".zip"
                elif "rar" in ct: ext = ".rar"
                elif "pdf" in ct: ext = ".pdf"
            return resp_name, ext or ".pdf"
    except Exception:
        resp_name = os.path.basename(urlparse(file_url).path) or "arquivo"
        ext = os.path.splitext(resp_name)[1].lower() or ".pdf"
        return resp_name, ext

# ---------- download with resume ----------
def download_with_resume(session, url, headers, dest_path, max_retries=6, chunk_size=128*1024, timeout=120):
    tmp = dest_path.with_suffix(dest_path.suffix + f".part.{uuid.uuid4().hex}")
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    total = None
    attempt = 0
    while attempt < max_retries:
        range_headers = dict(headers)
        if downloaded > 0:
            range_headers["Range"] = f"bytes={downloaded}-"
        try:
            with session.get(url, headers=range_headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                if downloaded > 0 and r.status_code == 200:
                    downloaded = 0
                    if tmp.exists(): os.unlink(_win_long(tmp))
                    attempt += 1; sleep(1.2 * attempt); continue
                cl = r.headers.get("Content-Length")
                if cl is not None:
                    part_len = int(cl)
                    total = (downloaded + part_len) if downloaded > 0 else part_len
                mode = "ab" if downloaded > 0 else "wb"
                with open(_win_long(tmp), mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk); downloaded += len(chunk)
                break
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                httplib.IncompleteRead):
            attempt += 1; sleep(1.2 * attempt); continue
    if total is not None and downloaded < total:
        raise IOError(f"Incomplete download: got {downloaded} of {total} bytes")
    os.replace(_win_long(tmp), _win_long(dest_path))
    return downloaded

# ---------- extraction ----------
def find_7z(explicit: str | None = None) -> str | None:
    if explicit and pathlib.Path(explicit).exists(): return explicit
    env = os.environ.get("SEVENZIP")
    if env and pathlib.Path(env).exists(): return env
    for c in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if pathlib.Path(c).exists(): return c
    return shutil.which("7z") or shutil.which("7z.exe") or shutil.which("7za")

def _extract_with_7z(src: pathlib.Path, out_dir: pathlib.Path) -> bool:
    sevenz = SEVENZ_PATH or find_7z(None)
    if not sevenz: return False
    try:
        cmd = [sevenz, "x", "-y", "-bb0", "-mcp=65001", f"-o{str(out_dir)}", str(src)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=480)
        return True
    except Exception:
        return False

def _extract_zip(src: pathlib.Path, out_dir: pathlib.Path) -> bool:
    try:
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(out_dir)
        return True
    except Exception:
        return False

def _extract_rar(src: pathlib.Path, out_dir: pathlib.Path) -> bool:
    try:
        with rarfile.RarFile(src, "r") as rf:
            rf.extractall(out_dir)
        return True
    except Exception:
        return False

def extract_nested_all_to(root: pathlib.Path, max_rounds: int = 20):
    rounds = 0
    while rounds < max_rounds:
        archives = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".zip", ".rar")]
        if not archives:
            break
        extracted_any = False
        for arch in archives:
            parent = arch.parent
            ok = _extract_with_7z(arch, parent)
            if not ok:
                ok = _extract_zip(arch, parent) if arch.suffix.lower()==".zip" else _extract_rar(arch, parent)
            if ok:
                extracted_any = True
                try: os.unlink(_win_long(arch))
                except Exception: pass
        if not extracted_any:
            break
        rounds += 1

def move_tree_skip_existing(src_root: pathlib.Path, dest_root: pathlib.Path):
    """Move tree from src_root to dest_root; skip files that already exist (long-path safe)."""
    os.makedirs(_win_long(dest_root), exist_ok=True)
    for d in sorted([p for p in src_root.rglob("*") if p.is_dir()]):
        rel = d.relative_to(src_root)
        os.makedirs(_win_long(dest_root / rel), exist_ok=True)
    for p in sorted([p for p in src_root.rglob("*") if p.is_file()]):
        rel = p.relative_to(src_root)
        dest = dest_root / rel
        os.makedirs(_win_long(dest.parent), exist_ok=True)
        if exists_long(dest):
            try: os.unlink(_win_long(p))
            except Exception: pass
            continue
        try:
            os.replace(_win_long(p), _win_long(dest))
        except Exception:
            shutil.copy2(_win_long(p), _win_long(dest))
            try: os.unlink(_win_long(p))
            except Exception: pass

# ---------- per-link dump with progress ----------
def dump_link(session, label: str, file_url: str, referer: str, dest_dir: pathlib.Path):
    t0 = time.perf_counter()
    _print(f"[start] {label} → {file_url}")

    headers = dict(HEADERS)
    if referer: headers["Referer"] = referer

    resp_name, ext = _sniff_name_and_ext(session, file_url, headers)

    # direct non-archive
    if ext not in (".zip",".rar"):
        os.makedirs(_win_long(dest_dir), exist_ok=True)
        safe_name = safe_segment(os.path.basename(resp_name) or "arquivo")
        dest = dest_dir / safe_name
        if not exists_long(dest):
            tmp = WORK_BASE / f"__tmp_{uuid.uuid4().hex}{ext or ''}"
            _ = download_with_resume(session, file_url, headers, tmp)
            try:
                os.replace(_win_long(tmp), _win_long(dest))
            except Exception:
                shutil.copy2(_win_long(tmp), _win_long(dest))
                try: os.unlink(_win_long(tmp))
                except Exception: pass
        dt = time.perf_counter() - t0
        _print(f"[done]  {label} (file) in {dt:0.1f}s")
        return

    # archive
    tmp_arch = WORK_BASE / f"__tmp_{uuid.uuid4().hex}{ext}"
    _ = download_with_resume(session, file_url, headers, tmp_arch)

    work = _short_workdir()
    ok = _extract_with_7z(tmp_arch, work)
    if not ok:
        if ext==".zip": ok = _extract_zip(tmp_arch, work)
        elif ext==".rar": ok = _extract_rar(tmp_arch, work)

    if not ok:
        os.makedirs(_win_long(dest_dir), exist_ok=True)
        raw_name = safe_segment(os.path.basename(resp_name) or tmp_arch.name)
        dest = dest_dir / raw_name
        if not exists_long(dest):
            try:
                shutil.copy2(_win_long(tmp_arch), _win_long(dest))
            except Exception as e:
                _print(f"[warn] failed to save raw archive for {label}: {e}")
        try: os.unlink(_win_long(tmp_arch))
        except Exception: pass
        dt = time.perf_counter() - t0
        _print(f"[warn] {label}: extract failed; saved raw archive ({dt:0.1f}s)")
        return

    extract_nested_all_to(work, max_rounds=20)
    move_tree_skip_existing(work, dest_dir)

    try: shutil.rmtree(work, ignore_errors=True)
    except Exception: pass
    try: os.unlink(_win_long(tmp_arch))
    except Exception: pass

    dt = time.perf_counter() - t0
    _print(f"[done]  {label} (archive) in {dt:0.1f}s")

# ---------- crawl with total timer ----------
def _norm_region_key(x: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (x or '').lower())

def coerce_to_download_page(x: str):
    """Accept slug or full URL. Return (slug, url)."""
    x = (x or "").strip()
    if not x:
        return None
    if x.lower().startswith(("http://","https://")):
        url  = x
        slug = os.path.splitext(os.path.basename(urlparse(url).path))[0]
    else:
        slug = os.path.splitext(os.path.basename(urlparse(x).path))[0]
        url  = f"{BASE}/downloads/{slug}.htm"
    return (slug, url)

def crawl(regions_filter=None, unis_filter=None, extra_pages=None, workers=4):
    t0_total = time.perf_counter()
    os.makedirs(_win_long(OUT_ROOT), exist_ok=True)
    s = make_session()

    # 1) Regions: only if user passed --regions
    if regions_filter:
        _all = collect_region_pages(s)
        want = {_norm_region_key(r) for r in regions_filter}
        regions = [(n,u) for (n,u) in _all if _norm_region_key(n) in want]
    else:
        regions = []

    # 2) Extra pages treated as regions
    extras = []
    if extra_pages:
        for x in extra_pages:
            cu = coerce_to_download_page(x)
            if cu:
                slug, url = cu
                slug = FRIENDLY_PAGE_SLUG.get(slug, slug)
                extras.append((slug, url))

    # Merge + de-dupe by URL
    if extras:
        seen = {u for (_,u) in regions}
        regions += [(n,u) for (n,u) in extras if u not in seen]

    if not regions:
        _print("No regions found."); return

    tasks = []
    for region_slug, region_url in regions:
        _print(f"[Region] {region_slug} -> {region_url}")
        uni_blocks = collect_universities_in_region(s, region_url)

        # If NO per-university pages, treat page as its own region folder
        if not uni_blocks:
            items_here = collect_items(s, region_url)
            if items_here:
                base_dir = OUT_ROOT / safe_segment(region_slug, "pagina")
                os.makedirs(_win_long(base_dir), exist_ok=True)
                _print(f"  [Direct] {region_slug} -> {len(items_here)} links")
                for label, url in items_here:
                    dest_dir = base_dir / safe_segment(label, "arquivo")
                    tasks.append((label, url, region_url, dest_dir))
                continue
            else:
                _print("  (no files on page)")
                continue

        # Normal flow: region page lists multiple universities
        if unis_filter:
            keys = [u.lower() for u in unis_filter]
            uni_blocks = [(n,u,a) for (n,u,a) in uni_blocks if any(k in n.lower() for k in keys)]
        if not uni_blocks:
            _print("  (no universities matched)")
            continue

        for uni_display_from_region, uni_url, uni_alias in uni_blocks:
            uni_dir   = safe_segment(uni_display_from_region, "universidade")
            alias_dir = safe_segment(uni_alias or uni_display_from_region, uni_dir)
            base_dir  = OUT_ROOT / region_slug / uni_dir / alias_dir
            os.makedirs(_win_long(base_dir), exist_ok=True)

            items = collect_items(s, uni_url)
            _print(f"  [University] {uni_display_from_region} (alias: {uni_alias}) -> {len(items)} links")
            if not items:
                _print("    (no files)")
                continue

            for label, url in items:
                dest_dir = base_dir / safe_segment(label, "arquivo")
                tasks.append((label, url, uni_url, dest_dir))

    if not tasks:
        _print("Nothing to do."); return

    _print(f"[Queue] {len(tasks)} items total\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(dump_link, s, label, url, referer, dest_dir)
                for (label, url, referer, dest_dir) in tasks]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                _print(f"[err] worker -> {e}")

    dt_total = time.perf_counter() - t0_total
    hh = int(dt_total // 3600); mm = int((dt_total % 3600) // 60); ss = int(dt_total % 60)
    _print(f"\nDone. Items processed: {len(tasks)} | Elapsed: {hh:02d}:{mm:02d}:{ss:02d}")

def parse_args():
    ap = argparse.ArgumentParser(description="Brasil Escola downloader (uni folder → alias subfolder → per-link dump)")
    ap.add_argument("--regions", nargs="*", help="Filter region slugs (e.g., sudeste | nordeste | centrooeste)")
    ap.add_argument("--unis", nargs="*", help="Filter university names (substring match)")
    ap.add_argument("--pages", nargs="*", help="Extra downloads pages (slug or full URL), treated as regions")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads (default 4)")
    ap.add_argument("--sevenzip", help="Path to 7z.exe (optional)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SEVENZ_PATH = find_7z(args.sevenzip)

    if SEVENZ_PATH:
        _print(f"[ok] Using 7-Zip at: {SEVENZ_PATH}")
    else:
        _print("[warn] 7-Zip not found. RAR extraction may be limited. Install 7-Zip or pass --sevenzip PATH.")

    crawl(
        regions_filter=set(args.regions) if args.regions else None,
        unis_filter=set(args.unis) if args.unis else None,
        extra_pages=list(args.pages) if args.pages else None,
        workers=args.workers,
    )


#python br_escola_downloader.py `
#  --regions centrooeste nordeste norte sudeste sul `
#  --pages olimpiada-brasileira-matematica-escolas-publicas `
#  --workers 6 `
#  --sevenzip "C:\Program Files\7-Zip\7z.exe"