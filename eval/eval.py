#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, tempfile, csv
import contextlib
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import lpips
import laion_clap
from tqdm import tqdm

# -----------------------------
# Basic utils
# -----------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=-1)

def safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")

def safe_std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else float("nan")

# -----------------------------
# Audio helpers
# -----------------------------
def chunk_audio(y: np.ndarray, sr: int, chunk_sec: float) -> list[np.ndarray]:
    """Split into fixed chunks (drop very short tail)."""
    if chunk_sec <= 0:
        return [y]
    n = int(chunk_sec * sr)
    if n <= 0 or len(y) <= n:
        return [y]
    chunks = []
    for s in range(0, len(y), n):
        c = y[s:s+n]
        if len(c) < n // 2:
            break
        chunks.append(c)
    return chunks if chunks else [y]

def _safe_leading_dash_filename(p: Path) -> Path:
    """Some datasets store ytid starting with '-' as filenames starting with '_'."""
    name = p.name
    if name.startswith("-"):
        name = "_" + name[1:]
    return p.with_name(name)

def resolve_src_wav(src_root: Path, wav_rel: str) -> Path:
    """
    wav_rel in JSON like 'segments_47s/-0SdAVK79lg_seg000.wav'
    But disk may have 'segments_47s/_0SdAVK79lg_seg000.wav'
    """
    p = src_root / wav_rel
    if p.exists():
        return p

    # try only filename leading dash -> underscore
    p2 = _safe_leading_dash_filename(p)
    if p2.exists():
        return p2

    # try also if parent folder exists but filename mismatch
    if p.parent.exists():
        cand = p.parent / _safe_leading_dash_filename(Path(p.name)).name
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Cannot find src wav: {p} (also tried {p2})")

def resolve_edit_wav(edit_dir: Path, src_wav: Path, edit_suffix: str, mirror_subdir: bool,
                     emphasize: str = "", name_mode: str = "suffix") -> Path:
    rel_parent = src_wav.parent.name if mirror_subdir else ""
    target_dir = edit_dir / rel_parent if mirror_subdir else edit_dir

    stem = src_wav.stem
    ext = src_wav.suffix

    em_raw = (emphasize or "").strip().lower()
    em_can = canon_attr(em_raw)

    def fname_attr_variants(s: str) -> list[str]:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)

        outs = []

        def add(x: str):
            x = (x or "").strip()
            if x and x not in outs:
                outs.append(x)

        # base forms
        bases = [s]

        # NEW: drop trailing " music"  (latin music -> latin)
        if s.endswith(" music"):
            bases.append(s[:-5].strip())  # remove " music"

        for b in bases:
            add(b)                   # "hip hop" / "latin"
            add(b.replace(" ", ""))  # "hiphop"
            add(b.replace(" ", "_")) # "hip_hop"
            add(b.replace(" ", "-")) # "hip-hop"

        return outs

    # filename matching variants (IMPORTANT: for matching only)
    em_vars = []
    for e in fname_attr_variants(em_raw) + fname_attr_variants(em_can):
        if e and e not in em_vars:
            em_vars.append(e)

    # special-case drum(s)
    if "drums" in em_vars and "drum" not in em_vars:
        em_vars.append("drum")
    if "drum" in em_vars and "drums" not in em_vars:
        em_vars.append("drums")

    # generic singular fallback
    for e in list(em_vars):
        if e.endswith("s") and len(e) > 1:
            es = e[:-1]
            if es and es not in em_vars:
                em_vars.append(es)

    # suffix variants: allow transformer, -transformer, _transformer
    suf0 = (edit_suffix or "")
    suffixes = [suf0]
    if suf0 and not suf0.startswith(("-", "_")):
        suffixes += ["-" + suf0, "_" + suf0]

    # handle leading '-' -> '_' stem variant
    stem_vars = [stem]
    if stem.startswith("-"):
        stem_vars.append("_" + stem[1:])

    cands = []

    if name_mode == "attr_suffix":
        # STRICT: MUST contain attribute token
        for st in stem_vars:
            for sep1 in ["_", "-"]:  # allow st_attr or st-attr
                for em in em_vars:
                    for suf in suffixes:
                        cands.append(target_dir / f"{st}{sep1}{em}{suf}{ext}")
    else:
        # suffix-only mode
        for st in stem_vars:
            for suf in suffixes:
                cands.append(target_dir / f"{st}{suf}{ext}")

        # keep old fallback only for suffix mode
        cands += [
            target_dir / f"{stem}_None{ext}",
            target_dir / f"{stem}_none{ext}",
        ]

    for c in cands:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Cannot find edited wav for {src_wav.name} in {target_dir}. "
        f"mode={name_mode} emphasize={emphasize} suffix='{edit_suffix}'. "
        f"Tried: {[str(x) for x in cands[:12]]}..."
    )

# -----------------------------
# CLAP prompt builder (STRICT, based on stats/emphasize)
# -----------------------------
GENRE_HINTS = {
    "rock","jazz","country","blues","electronic","latin","latin music","hip hop","hiphop",
    "pop","soul","classical","metal","reggae","funk","rnb","r&b","edm","techno","house",
    "chinese","kpop","k-pop","j-pop","jpop"
}

SYNONYM = {
    "drum": "drums",
    "percussive drums": "drums",
    "harps": "harp",
    "blue": "blues",
    " country": "country",
}

def canon_attr(x: str) -> str:
    x = (x or "").strip().lower()
    x = SYNONYM.get(x, x)
    x = re.sub(r"\s+", " ", x)
    return x

def infer_kind_from_stats(stats_dict: dict) -> str:
    """
    Decide whether this editing_type_id corresponds to 'genre' or 'instrument'
    by looking at how many keys look like genres.
    """
    keys = [canon_attr(k) for k in stats_dict.keys()]
    if not keys:
        return "attribute"
    hit = sum(1 for k in keys if k in GENRE_HINTS)
    # if many keys are genre-like, treat as genre
    return "genre" if hit >= max(2, int(0.3 * len(keys))) else "instrument"

def strict_prompts_for(attr: str, kind: str) -> list[str]:
    """
    Make prompts "strict": attribute must be dominant/foreground.
    """
    a = canon_attr(attr)
    if not a:
        return []

    if kind == "genre":
        # keep them short & decisive
        return [
            f"this is {a} music",
            f"a {a} track",
            f"a song in the {a} genre",
            f"{a} style music",
        ]

    # instrument (default)
    return [
        f"music featuring {a} as the dominant instrument",
        f"the main instrument is {a}",
        f"{a} is clearly audible and in the foreground",
        f"a recording with prominent {a} leading the arrangement",
    ]

def build_prompt_bank(data: dict) -> dict:
    """
    Returns: prompt_bank[editing_type_id][attr] = [prompts...]
    Uses stats[...] keys to build prompts for ALL attributes in stats.
    """
    stats = data.get("stats", {})
    bank = {}
    for etid, d in stats.items():
        kind = infer_kind_from_stats(d if isinstance(d, dict) else {})
        bank[str(etid)] = {}
        if isinstance(d, dict):
            for k in d.keys():
                a = canon_attr(k)
                bank[str(etid)][a] = strict_prompts_for(a, kind)
    return bank

BRACKET_RE = re.compile(r"\[([^\]]+)\]")

def extract_bracket_terms(s: str) -> list[str]:
    if not s:
        return []
    return [t.strip() for t in BRACKET_RE.findall(s) if t.strip()]

def attr_from_prompt(prompt: str, fallback: str = "") -> str:
    terms = extract_bracket_terms(prompt)
    if terms:
        return " and ".join(terms)
    return (fallback or "").strip()

def dedup_prompts(prompts: list[str]) -> list[str]:
    seen = set()
    out = []
    for p in prompts:
        p = p.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out

def build_target_prompts(data: dict, prompt_bank: dict, etid: str, target_attr: str) -> tuple[str, list[str]]:
    target_attr = canon_attr(target_attr)
    bank_for_etid = prompt_bank.get(str(etid), {})
    if target_attr in bank_for_etid:
        return target_attr, list(bank_for_etid[target_attr])

    kind = infer_kind_from_stats(data.get("stats", {}).get(str(etid), {}) or {})
    return target_attr, strict_prompts_for(target_attr, kind)

def build_original_prompts(data: dict, etid: str, original_prompt: str) -> tuple[str, list[str]]:
    original_attr = canon_attr(attr_from_prompt(original_prompt, fallback=""))
    if not original_attr:
        return "", []

    kind = infer_kind_from_stats(data.get("stats", {}).get(str(etid), {}) or {})
    return original_attr, strict_prompts_for(original_attr, kind)

def fmt_metric(v: float) -> str:
    return "" if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{v:.6f}"


# -----------------------------
# CLAP (LAION-CLAP) with chunking + multi-prompt
# -----------------------------
CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_URL = f"https://huggingface.co/lukewys/laion_clap/resolve/main/{CLAP_CKPT_NAME}"


def download_clap_checkpoint(out_path: Path) -> Path:
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[INFO] CLAP checkpoint not found. Downloading {CLAP_CKPT_NAME}...")
    print(f"[INFO] Source: {CLAP_CKPT_URL}")
    print(f"[INFO] Save to: {out_path}")

    with urllib.request.urlopen(CLAP_CKPT_URL) as response:
        total = int(response.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        last_report = 0

        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct >= last_report + 5:
                        print(f"[INFO] Downloaded {pct}%")
                        last_report = pct

    tmp_path.replace(out_path)
    print(f"[INFO] Checkpoint ready: {out_path}")
    return out_path


def resolve_clap_checkpoint() -> Path:
    env_path = os.environ.get("MELODIA_CLAP_CKPT", "").strip()
    if env_path:
        env_ckpt = Path(env_path)
        if env_ckpt.exists():
            return env_ckpt
        return download_clap_checkpoint(env_ckpt)

    here = Path(__file__).resolve().parent
    candidates = [
        here / CLAP_CKPT_NAME,
        here.parent / CLAP_CKPT_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path

    return download_clap_checkpoint(candidates[0])


class ClapScorer:
    def __init__(self, device: str = "cuda", clap_sr: int = 48000, chunk_sec: float = 10.0,
                 map01: bool = False, prompt_reduce: str = "mean"):
        self.device = device
        self.clap_sr = clap_sr
        self.chunk_sec = chunk_sec
        self.map01 = map01
        self.prompt_reduce = prompt_reduce

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        ckpt_path = resolve_clap_checkpoint()
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model.load_ckpt(str(ckpt_path))

        # cache text embeddings: key = tuple(prompts)
        self._text_cache = {}

    def _post(self, cos: torch.Tensor) -> torch.Tensor:
        return (cos + 1.0) / 2.0 if self.map01 else cos

    @torch.no_grad()
    def _get_text_emb(self, prompts: list[str]) -> torch.Tensor:
        key = tuple(prompts)
        if key in self._text_cache:
            return self._text_cache[key]
        t = self.model.get_text_embedding(prompts, use_tensor=True)  # [K, D]
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        self._text_cache[key] = t
        return t

    @torch.no_grad()
    def audio_emb(self, wav_path: Path) -> torch.Tensor:
        y, _ = librosa.load(str(wav_path), sr=self.clap_sr, mono=True)
        chunks = chunk_audio(y, self.clap_sr, self.chunk_sec)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            chunk_paths = []
            for i, c in enumerate(chunks):
                tmp_wav = td / f"chunk_{i:03d}.wav"
                sf.write(str(tmp_wav), c, self.clap_sr)
                chunk_paths.append(str(tmp_wav))

            a = self.model.get_audio_embedding_from_filelist(chunk_paths, use_tensor=True)  # [N, D]
            a0 = a.mean(dim=0)  # [D]
            a0 = a0 / (a0.norm(dim=-1, keepdim=True) + 1e-8)
        return a0

    @torch.no_grad()
    def score_from_emb(self, audio_emb: torch.Tensor, prompts) -> float:
        prompts = list(prompts) if isinstance(prompts, (list, tuple)) else [str(prompts)]
        prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        if not prompts:
            return float("nan")

        t = self._get_text_emb(prompts)  # [K, D]
        a0 = audio_emb / (audio_emb.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (t * a0).sum(dim=-1)  # [K]
        sim = self._post(cos)
        return float(sim.max().item()) if self.prompt_reduce == "max" else float(sim.mean().item())

    @torch.no_grad()
    def score_multi(self, wav_path: Path, prompts) -> float:
        return self.score_from_emb(self.audio_emb(wav_path), prompts)

# -----------------------------
# LPIPS on log-mel patches (old args keep the lpaps_ prefix)
# -----------------------------
class LpapsApprox:
    def __init__(self, device: str = "cuda", sr: int = 16000, n_mels: int = 128,
                 patch_sec: float = 2.0, hop_sec: float = 2.0, reduce: str = "sum"):
        self.device = device
        self.sr = sr
        self.n_mels = n_mels
        self.patch_sec = patch_sec
        self.hop_sec = hop_sec
        self.reduce = reduce

        self.n_fft = 1024
        self.hop_length = 256  # keep same as you used

        self.loss = lpips.LPIPS(net="vgg").to(device)
        self.loss.eval()

    def _mel_db(self, y: np.ndarray) -> np.ndarray:
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        S = librosa.power_to_db(S, ref=np.max)   # [M, T]
        return S.astype(np.float32)

    def _patch_to_img(self, P: np.ndarray) -> torch.Tensor:
        # normalize this patch to [-1,1] (keep your original behavior)
        P = (P - P.min()) / (P.max() - P.min() + 1e-8)
        P = (P * 2.0 - 1.0).astype(np.float32)
        img = torch.from_numpy(P)[None, None, :, :].repeat(1, 3, 1, 1).to(self.device)
        return img

    @torch.no_grad()
    def score_from_mel(self, S1: np.ndarray, S2: np.ndarray) -> float:
        T = min(S1.shape[1], S2.shape[1])
        if T <= 0:
            return 0.0
        S1 = S1[:, :T]
        S2 = S2[:, :T]

        win_frames = max(1, int(self.patch_sec * self.sr / self.hop_length))
        hop_frames = max(1, int(self.hop_sec * self.sr / self.hop_length))

        vals = []
        for f0 in range(0, max(1, T - win_frames + 1), hop_frames):
            P1 = S1[:, f0:f0 + win_frames]
            P2 = S2[:, f0:f0 + win_frames]
            a = self._patch_to_img(P1)
            b = self._patch_to_img(P2)
            vals.append(self.loss(a, b).mean().item())

        if not vals:
            return 0.0
        return float(np.sum(vals) if self.reduce == "sum" else np.mean(vals))

    @torch.no_grad()
    def score(self, src_wav: Path, edit_wav: Path) -> float:
        y1, _ = librosa.load(str(src_wav), sr=self.sr, mono=True)
        y2, _ = librosa.load(str(edit_wav), sr=self.sr, mono=True)
        L = min(len(y1), len(y2))
        if L <= 0:
            return 0.0
        y1, y2 = y1[:L], y2[:L]

        S1 = self._mel_db(y1)
        S2 = self._mel_db(y2)
        return self.score_from_mel(S1, S2)

# -----------------------------
# Chroma similarity (quantized top pitch class)
# -----------------------------
def chroma_similarity(src_wav: Path, edit_wav: Path, sr: int = 22050, hop: int = 512) -> float:
    y1, _ = librosa.load(str(src_wav), sr=sr, mono=True)
    y2, _ = librosa.load(str(edit_wav), sr=sr, mono=True)

    C1 = librosa.feature.chroma_cqt(y=y1, sr=sr, hop_length=hop)
    C2 = librosa.feature.chroma_cqt(y=y2, sr=sr, hop_length=hop)

    T = min(C1.shape[1], C2.shape[1])
    if T == 0:
        return 0.0
    i1 = np.argmax(C1[:, :T], axis=0)
    i2 = np.argmax(C2[:, :T], axis=0)
    return float((i1 == i2).mean())


# -----------------------------
# CSV writer
# -----------------------------
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # dataset json (your format)
    ap.add_argument("--task_json", required=True, type=str,
                    help="JSON containing {stats, top5_emphasize, segments:[...]}")

    # roots
    ap.add_argument("--src_root", required=True, type=str, help="Root that contains segments_47s/ ...")
    ap.add_argument("--method", action="append", required=True, help="name=DIR (repeatable). e.g. --method edit=/path/to/muse_prompt")
    ap.add_argument("--edit_suffix", default="_None", type=str, help="suffix appended to src stem for edited wavs (default: _None)")
    ap.add_argument("--mirror_subdir", action="store_true",
                    help="If set, keep the relative parent folder name when searching edited wavs (rarely needed).")

    # output
    ap.add_argument("--out_csv", required=True, type=str, help="per-segment metrics CSV")
    ap.add_argument("--out_summary_csv", default="", type=str, help="per-method summary CSV (default: out_csv with _summary suffix)")

    # CLAP knobs
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--clap_sr", type=int, default=48000)
    ap.add_argument("--clap_chunk_sec", type=float, default=10.0)
    ap.add_argument("--clap_map01", action="store_true")
    ap.add_argument("--clap_prompt_reduce", choices=["mean", "max"], default="max")

    # LPIPS knobs (argument names keep the old lpaps_ prefix)
    ap.add_argument("--lpaps_sr", type=int, default=16000)
    ap.add_argument("--lpaps_patch_sec", type=float, default=2.0)
    ap.add_argument("--lpaps_hop_sec", type=float, default=2.0)
    ap.add_argument("--lpaps_reduce", choices=["sum", "mean"], default="sum")

    # prompts control
    ap.add_argument("--use_json_editing_prompt", action="store_true",
                    help="Also include segment['editing_prompt'] as an extra CLAP prompt (optional).")
    ap.add_argument("--joiner", default=" || ", type=str, help="Prompt joiner stored in CSV.")
    ap.add_argument("--edit_name_mode", choices=["suffix", "attr_suffix"], default="suffix")


    args = ap.parse_args()

    src_root = Path(args.src_root)
    task_path = Path(args.task_json)
    data = json.loads(task_path.read_text(encoding="utf-8"))

    segments = data.get("segments", [])
    if not isinstance(segments, list) or len(segments) == 0:
        raise SystemExit("task_json has no segments[]")

    prompt_bank = build_prompt_bank(data)

    # parse methods
    methods = []
    for m in args.method:
        name, d = m.split("=", 1)
        methods.append((name, Path(d)))

    clap = ClapScorer(
        device=args.device,
        clap_sr=args.clap_sr,
        chunk_sec=args.clap_chunk_sec,
        map01=args.clap_map01,
        prompt_reduce=args.clap_prompt_reduce,
    )
    lpaps = LpapsApprox(
        device=args.device,
        sr=args.lpaps_sr,
        patch_sec=args.lpaps_patch_sec,
        hop_sec=args.lpaps_hop_sec,
        reduce=args.lpaps_reduce,
    )

    # per-segment rows
    seg_rows = []

    metric_keys = [
        "CLAP",
        "src_CLAP_target",
        "src_CLAP_original",
        "edit_CLAP_target",
        "edit_CLAP_original",
        "src_GAP",
        "edit_GAP",
        "GAP_delta",
        "delta_target",
        "delta_away_original",
        "LPIPS",
        "Chroma",
    ]

    # per-method accumulators
    per_method = {name: {"n": 0, **{k: [] for k in metric_keys}} for name, _ in methods}

    for idx, seg in enumerate(tqdm(segments)):
        # basic info
        ytid = seg.get("ytid", "")
        segment_id = seg.get("segment_id", "")
        etid = str(seg.get("editing_type_id", ""))
        emphasize_raw = seg.get("emphasize", "")


        wav_rel = seg.get("wav", "")
        if not wav_rel:
            continue

        # resolve src wav on disk
        try:
            src_wav = resolve_src_wav(src_root, wav_rel)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        target_attr, prompts_target = build_target_prompts(data, prompt_bank, etid, emphasize_raw)
        original_attr, prompts_original = build_original_prompts(data, etid, seg.get("original_prompt", ""))

        # optionally include json's editing_prompt (can be too specific; use only if you want)
        if args.use_json_editing_prompt:
            ep = seg.get("editing_prompt", "")
            if isinstance(ep, str) and ep.strip():
                prompts_target.append(re.sub(r"\[|\]", "", ep))

        prompts_target = dedup_prompts(prompts_target)
        prompts_original = dedup_prompts(prompts_original)

        src_emb = clap.audio_emb(src_wav)
        src_clap_target = clap.score_from_emb(src_emb, prompts_target)
        src_clap_original = clap.score_from_emb(src_emb, prompts_original)
        src_gap = (
            src_clap_target - src_clap_original
            if np.isfinite(src_clap_target) and np.isfinite(src_clap_original)
            else float("nan")
        )

        for method_name, method_dir in methods:
            try:
                edit_wav = resolve_edit_wav(
                    method_dir, src_wav, args.edit_suffix, args.mirror_subdir,
                    emphasize=emphasize_raw, name_mode=args.edit_name_mode
                )
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue

            # metrics
            edit_emb = clap.audio_emb(edit_wav)
            edit_clap_target = clap.score_from_emb(edit_emb, prompts_target)
            edit_clap_original = clap.score_from_emb(edit_emb, prompts_original)
            edit_gap = (
                edit_clap_target - edit_clap_original
                if np.isfinite(edit_clap_target) and np.isfinite(edit_clap_original)
                else float("nan")
            )
            gap_delta = (
                edit_gap - src_gap
                if np.isfinite(edit_gap) and np.isfinite(src_gap)
                else float("nan")
            )
            delta_target = (
                edit_clap_target - src_clap_target
                if np.isfinite(edit_clap_target) and np.isfinite(src_clap_target)
                else float("nan")
            )
            delta_away_original = (
                src_clap_original - edit_clap_original
                if np.isfinite(src_clap_original) and np.isfinite(edit_clap_original)
                else float("nan")
            )
            s_lpips = lpaps.score(src_wav, edit_wav)
            s_chroma = chroma_similarity(src_wav, edit_wav)

            row_metrics = {
                "CLAP": edit_clap_target,
                "src_CLAP_target": src_clap_target,
                "src_CLAP_original": src_clap_original,
                "edit_CLAP_target": edit_clap_target,
                "edit_CLAP_original": edit_clap_original,
                "src_GAP": src_gap,
                "edit_GAP": edit_gap,
                "GAP_delta": gap_delta,
                "delta_target": delta_target,
                "delta_away_original": delta_away_original,
                "LPIPS": s_lpips,
                "Chroma": s_chroma,
            }

            per_method[method_name]["n"] += 1
            for key, value in row_metrics.items():
                if np.isfinite(value):
                    per_method[method_name][key].append(float(value))

            seg_rows.append({
                "method": method_name,
                "ytid": ytid,
                "segment_id": segment_id,
                "editing_type_id": etid,
                "target_attr": target_attr,
                "original_attr": original_attr,
                "emphasize": target_attr,
                "src_wav": str(src_wav),
                "edit_wav": str(edit_wav),
                **{k: fmt_metric(v) for k, v in row_metrics.items()},
                "prompts_target": args.joiner.join(prompts_target),
                "prompts_original": args.joiner.join(prompts_original),
                "json_editing_prompt": seg.get("editing_prompt", ""),
                "json_original_prompt": seg.get("original_prompt", ""),
            })

        # if (idx + 1) % 50 == 0:
        #     print(f"[Info] processed segments: {idx+1}/{len(segments)}")

    if len(seg_rows) == 0:
        raise SystemExit("No segment rows computed. Check paths/suffix and task_json wav fields.")

    # write per-segment CSV
    out_csv = Path(args.out_csv)
    seg_fields = [
        "method","ytid","segment_id","editing_type_id","target_attr","original_attr","emphasize",
        "src_wav","edit_wav",
        "CLAP","src_CLAP_target","src_CLAP_original","edit_CLAP_target","edit_CLAP_original",
        "src_GAP","edit_GAP","GAP_delta","delta_target","delta_away_original",
        "LPIPS","Chroma","prompts_target","prompts_original",
        "json_editing_prompt","json_original_prompt"
    ]
    write_csv(out_csv, seg_rows, seg_fields)

    # per-method summary
    summary_rows = []
    for name, _ in methods:
        n = per_method[name]["n"]
        if n <= 0:
            continue
        row = {
            "method": name,
            "n_pairs": n,
        }
        for key in metric_keys:
            row[f"{key}_mean"] = safe_mean(per_method[name][key])
            row[f"{key}_std"] = safe_std(per_method[name][key])
        summary_rows.append(row)

    # write summary CSV
    if args.out_summary_csv.strip():
        out_summary = Path(args.out_summary_csv)
    else:
        out_summary = out_csv.with_name(out_csv.stem + "_summary.csv")

    sum_fields = ["method","n_pairs"]
    for key in metric_keys:
        sum_fields += [f"{key}_mean", f"{key}_std"]

    sum_rows_str = []
    for r in summary_rows:
        rr = dict(r)
        for k in sum_fields:
            v = rr[k]
            if isinstance(v, float):
                rr[k] = fmt_metric(v)
        sum_rows_str.append(rr)
    write_csv(out_summary, sum_rows_str, sum_fields)

    # print summary
    print("=== Summary (mean over pairs) ===")
    for r in summary_rows:
        print(
            f"{r['method']:>12} | "
            f"CLAP {r['CLAP_mean']:.4f} | "
            f"LPIPS {r['LPIPS_mean']:.4f} | "
            f"Chroma {r['Chroma_mean']:.4f} | "
            f"src_CLAP_target {r['src_CLAP_target_mean']:.4f} | "
            f"edit_CLAP_original {r['edit_CLAP_original_mean']:.4f}"
        )

    print(f"[Saved] per-segment: {out_csv}")
    print(f"[Saved] summary    : {out_summary}")

if __name__ == "__main__":
    main()
