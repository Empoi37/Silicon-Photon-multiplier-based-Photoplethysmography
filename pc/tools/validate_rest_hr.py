#!/usr/bin/env python3
"""Compare the prototype's live BPM (as recorded by the GUI) to a reference
heart-rate log (smartwatch, chest strap, manual readings) and produce the
MAE/bias/quantile statistics needed for Rapport.pdf, tableaux 4.1 et 4.3, et
la figure 4.1.

Ce script ne fait AUCUN retraitement du signal : il reprend directement les
colonnes ``bpm`` / ``fft_quality`` telles qu'enregistrées en direct par la
fenêtre principale (``app/main_window.py``, colonne ``bpm`` = sortie finale
de PpgHrProcessor.compute(), correction ML incluse). C'est donc une mesure
de ce que l'utilisateur voyait réellement à l'écran, pas une reconstruction
a posteriori.

Format du fichier CSV d'enregistrement GUI
-------------------------------------------
Produit par le bouton "Start CSV recording" du panneau Enregistrement.
Colonnes utilisées : unix_time_s, bpm (vide si invalide), fft_quality,
motion_level.

Format du fichier de référence (une ligne par lecture de la montre)
---------------------------------------------------------------------
CSV avec une ligne d'en-tête et deux colonnes :

    unix_time_s,bpm_ref        (recommandé : horodatage epoch, ex. time.time()
                                 noté au moment de lire la montre)
ou
    elapsed_s,bpm_ref           (secondes écoulées depuis le DÉBUT de
                                 l'enregistrement GUI -- démarrer le


                                 chronomètre de référence exactement au
                                 moment où l'on clique "Start CSV recording")

Usage
-----
  # Un seul sujet :
  python tools/validate_rest_hr.py --pair recording.csv reference.csv \
      --label "Sujet 1" --plot rapport/fig_4_1_sujet1.png

  # Plusieurs sujets agrégés pour le tableau 4.1 :
  python tools/validate_rest_hr.py \
      --pair rec_s1.csv ref_s1.csv --label "Sujet 1" \
      --pair rec_s2.csv ref_s2.csv --label "Sujet 2" \
      --pair rec_s3.csv ref_s3.csv --label "Sujet 3" \
      --plot rapport/fig_4_1.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Recording:
    label: str
    t: np.ndarray            # unix_time_s, une valeur par échantillon (250 Hz)
    bpm: np.ndarray           # NaN où invalide
    fft_quality: np.ndarray
    motion_level: np.ndarray


@dataclass
class Reference:
    t: np.ndarray             # unix_time_s (converti si le fichier était en elapsed_s)
    bpm: np.ndarray


@dataclass
class MatchedPoint:
    t: float
    bpm_meas: float
    bpm_ref: float
    dt_s: float  # écart temporel réel entre la lecture de référence et l'échantillon apparié


def load_recording(path: Path, label: str) -> Recording:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"Enregistrement vide : {path}")
    data = np.atleast_1d(data)
    names = [n.lower() for n in data.dtype.names]
    if "unix_time_s" not in names or "bpm" not in names:
        raise ValueError(
            f"{path} ne ressemble pas à un enregistrement GUI "
            f"(colonnes trouvées : {names})"
        )

    t = data["unix_time_s"].astype(float)

    def to_float_or_nan(x) -> float:
        s = x.decode() if isinstance(x, bytes) else str(x)
        s = s.strip()
        if s == "":
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    bpm_raw = data["bpm"]
    bpm = np.array([to_float_or_nan(v) for v in bpm_raw], dtype=float)

    fft_q = data["fft_quality"] if "fft_quality" in names else np.full_like(t, np.nan)
    fft_q = np.array(
        [to_float_or_nan(v) for v in fft_q], dtype=float
    ) if fft_q.dtype.kind not in "fc" else fft_q.astype(float)

    motion = data["motion_level"].astype(float) if "motion_level" in names else np.full_like(t, np.nan)

    return Recording(label=label, t=t, bpm=bpm, fft_quality=fft_q, motion_level=motion)


def load_reference(path: Path, recording_start_unix: float) -> Reference:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"Référence vide : {path}")
    data = np.atleast_1d(data)
    names = [n.lower() for n in data.dtype.names]

    bpm_col = next((c for c in ("bpm_ref", "bpm") if c in names), None)
    if bpm_col is None:
        raise ValueError(f"{path} : colonne bpm_ref (ou bpm) introuvable (colonnes: {names})")
    bpm = data[bpm_col].astype(float)

    if "unix_time_s" in names:
        t = data["unix_time_s"].astype(float)
    elif "elapsed_s" in names or "t_s" in names:
        col = "elapsed_s" if "elapsed_s" in names else "t_s"
        t = recording_start_unix + data[col].astype(float)
    else:
        raise ValueError(
            f"{path} : attendu une colonne 'unix_time_s' ou 'elapsed_s' "
            f"(colonnes trouvées : {names})"
        )
    return Reference(t=t, bpm=bpm)


def align(rec: Recording, ref: Reference, tol_s: float) -> tuple[list[MatchedPoint], int]:
    """Apparie chaque lecture de référence à l'échantillon mesuré valide le
    plus proche dans le temps, à condition que l'écart reste sous tol_s.
    Retourne les points appariés et le nombre de lectures de référence
    ignorées faute d'échantillon valide assez proche."""
    valid_mask = np.isfinite(rec.bpm)
    t_valid = rec.t[valid_mask]
    bpm_valid = rec.bpm[valid_mask]

    matches: list[MatchedPoint] = []
    skipped = 0
    if len(t_valid) == 0:
        return matches, len(ref.t)

    for t_r, b_r in zip(ref.t, ref.bpm):
        idx = int(np.argmin(np.abs(t_valid - t_r)))
        dt = float(t_valid[idx] - t_r)
        if abs(dt) > tol_s:
            skipped += 1
            continue
        matches.append(MatchedPoint(t=t_r, bpm_meas=float(bpm_valid[idx]), bpm_ref=float(b_r), dt_s=dt))
    return matches, skipped


def stats_from_matches(matches: list[MatchedPoint]) -> dict:
    if not matches:
        return {}
    err = np.array([m.bpm_meas - m.bpm_ref for m in matches])
    abs_err = np.abs(err)
    return {
        "n": len(matches),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "std": float(np.std(err)),
        "median_ae": float(np.median(abs_err)),
        "p10_ae": float(np.percentile(abs_err, 10)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "max_ae": float(np.max(abs_err)),
    }


def invalid_fraction(rec: Recording) -> float:
    if len(rec.bpm) == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(rec.bpm)))


def duration_s(rec: Recording) -> float:
    if len(rec.t) < 2:
        return 0.0
    return float(rec.t[-1] - rec.t[0])


def lock_delay_s(rec: Recording) -> float:
    """Temps entre le début de l'enregistrement (rec.t[0]) et le premier
    échantillon `bpm` valide -- délai de première mesure stable (§4.8).
    NaN si aucun échantillon valide."""
    valid = np.isfinite(rec.bpm)
    if not np.any(valid):
        return float("nan")
    idx = int(np.argmax(valid))  # index du premier True
    return float(rec.t[idx] - rec.t[0])


def print_subject_row(label: str, rec: Recording, matches: list[MatchedPoint], skipped: int):
    st = stats_from_matches(matches)
    inv = invalid_fraction(rec) * 100.0
    dur_min = duration_s(rec) / 60.0
    delay = lock_delay_s(rec)
    delay_str = f"{delay:.1f} s" if np.isfinite(delay) else "n/a"
    if not st:
        print(f"| {label} | {dur_min:.1f} min | {delay_str} | 0 (aucun appariement) | "
              f"-- | -- | -- | -- | -- | {inv:.1f} % |")
        return
    print(
        f"| {label} | {dur_min:.1f} min | {delay_str} | n={st['n']} (skip={skipped}) | "
        f"MAE={st['mae']:.2f} BPM | biais={st['bias']:+.2f} | "
        f"écart-type={st['std']:.2f} | médiane={st['median_ae']:.2f} | "
        f"P10-P90={st['p10_ae']:.2f}-{st['p90_ae']:.2f} | invalides={inv:.1f} % |"
    )


def plot_comparison(rec: Recording, ref: Reference, matches: list[MatchedPoint], out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = rec.t[0]
    t_rec = rec.t - t0
    bpm = rec.bpm

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Zones invalides (fond gris clair), fusion des segments contigus.
    invalid = ~np.isfinite(bpm)
    if np.any(invalid):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], invalid.astype(int), [0]))))
        for start, end in zip(edges[0::2], edges[1::2]):
            ax.axvspan(t_rec[start], t_rec[min(end, len(t_rec) - 1)],
                       color="#cccccc", alpha=0.4, linewidth=0, zorder=0)

    delay = lock_delay_s(rec)
    if np.isfinite(delay):
        ax.axvline(delay, color="#2c7a2c", linestyle="--", linewidth=1.2, zorder=2,
                   label=f"Premier verrouillage BPM (Δt={delay:.1f} s)")

    ax.plot(t_rec, bpm, color="#1f4e79", linewidth=1.3, label="BPM prototype (affiché en direct)")

    ref_t_rel = ref.t - t0
    ax.scatter(ref_t_rel, ref.bpm, color="#d95f02", s=28, zorder=3, label="BPM montre (référence)")

    for m in matches:
        ax.plot([m.t - t0, m.t - t0], [m.bpm_ref, m.bpm_meas],
                color="#999999", linewidth=0.6, zorder=1)

    ax.set_xlabel("Temps depuis le début de l'enregistrement (s)")
    ax.set_ylabel("Rythme cardiaque (BPM)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figure enregistrée : {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, metavar=("RECORDING_CSV", "REFERENCE_CSV"),
                     action="append", required=True,
                     help="Paire enregistrement GUI / référence montre. Répéter --pair par sujet.")
    ap.add_argument("--label", action="append", default=[],
                     help="Étiquette du sujet pour chaque --pair, dans le même ordre.")
    ap.add_argument("--tol", type=float, default=5.0,
                     help="Tolérance d'appariement temporel en secondes (défaut: 5 s).")
    ap.add_argument("--plot", type=Path, default=None,
                     help="Chemin PNG pour la figure 4.1 (un seul sujet à la fois).")
    args = ap.parse_args()

    labels = args.label + [f"Sujet {i + 1}" for i in range(len(args.label), len(args.pair))]

    all_matches: list[MatchedPoint] = []
    all_durations = []
    all_invalid_fracs = []
    all_delays = []
    n_subjects = len(args.pair)

    print(f"# Validation MAE au repos -- {n_subjects} sujet(s), tolérance ±{args.tol:.0f} s\n")
    print("| Sujet | Durée | Délai verrouillage | Points appariés | MAE | Biais | Écart-type | Médiane |AE| | P10-P90 |AE| | Invalides |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    last_rec = last_ref = last_matches = None
    for (rec_path, ref_path), label in zip(args.pair, labels):
        rec = load_recording(Path(rec_path), label)
        ref = load_reference(Path(ref_path), rec.t[0])
        matches, skipped = align(rec, ref, args.tol)
        print_subject_row(label, rec, matches, skipped)
        all_matches.extend(matches)
        all_durations.append(duration_s(rec))
        all_invalid_fracs.append(invalid_fraction(rec))
        all_delays.append(lock_delay_s(rec))
        last_rec, last_ref, last_matches = rec, ref, matches

    if n_subjects > 1:
        agg = stats_from_matches(all_matches)
        if agg:
            print(
                f"| **Agrégat ({n_subjects} sujets)** | "
                f"{sum(all_durations) / 60.0:.1f} min cumulées | -- | n={agg['n']} | "
                f"**MAE={agg['mae']:.2f} BPM** | biais={agg['bias']:+.2f} | "
                f"écart-type={agg['std']:.2f} | médiane={agg['median_ae']:.2f} | "
                f"P10-P90={agg['p10_ae']:.2f}-{agg['p90_ae']:.2f} | "
                f"invalides={100.0 * np.mean(all_invalid_fracs):.1f} % |"
            )

    finite_delays = [d for d in all_delays if np.isfinite(d)]
    if finite_delays:
        print(
            f"\nDélai de verrouillage BPM (temps entre le début de l'enregistrement et le "
            f"premier échantillon `bpm` valide), {len(finite_delays)} sujet(s) : "
            f"min={min(finite_delays):.1f} s | médiane={float(np.median(finite_delays)):.1f} s | "
            f"max={max(finite_delays):.1f} s -- alimente le Tableau 4.2, ligne « Première "
            f"mesure stable » (§4.8), à la place du chronométrage manuel."
        )

    print(
        "\nCes valeurs alimentent directement le Tableau 4.1 (N sujets, durée, MAE, "
        "invalides), le Tableau 4.3 (verdict MAE repos <= 3 BPM) et la conclusion "
        "(chapitre 8). Reporter aussi le biais et l'écart-type dans le texte : le "
        "rapport exige explicitement ces valeurs en plus de la MAE (section 4.1)."
    )

    if args.plot and last_rec is not None:
        if n_subjects > 1:
            print(
                "\nAvertissement : --plot ne trace que le DERNIER sujet de la liste "
                "--pair. Relancer une fois par sujet pour obtenir une figure par sujet, "
                "ou en choisir un représentatif pour la figure 4.1.",
                file=sys.stderr,
            )
        plot_comparison(last_rec, last_ref, last_matches, args.plot,
                         title=f"Comparaison temporelle -- {last_rec.label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
