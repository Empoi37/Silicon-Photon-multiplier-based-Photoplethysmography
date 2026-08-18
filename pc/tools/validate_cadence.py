#!/usr/bin/env python3
"""Compute cadence statistics (mean, median, P1/P99, probable gaps) for the
USB and BLE rows of Rapport.pdf, tableau 4.2.

Réutilise les chargeurs de tools/visualize_log.py (déjà capables de lire un
enregistrement GUI CSV ou un export nRF Connect) pour éviter de dupliquer le
parseur BLE.

Portée honnête (voir Annexe A du rapport) : le firmware ne transmet ni
compteur de séquence ni horodatage matériel. Ce script mesure donc la cadence
d'ARRIVÉE côté hôte (PC pour l'USB, téléphone pour le BLE) -- il ne peut pas
distinguer avec certitude une perte d'échantillon d'une simple gigue de
réception, ni détecter un réordonnancement. Les "sauts probables" rapportés
sont une heuristique (intervalle > 1.5x la période nominale), pas un compteur
de pertes garanti. Pour une mesure certaine, il faut le compteur de séquence
côté firmware recommandé en section 8.1, point 9.

Usage
-----
  python tools/validate_cadence.py data/recordings/ppg_raw_minimal_usb.csv \
      --label "USB minimal" --nominal-hz 250

  python tools/validate_cadence.py nrf_connect_export.csv \
      --label "BLE" --nominal-hz 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_log import load_log  # noqa: E402  (chemin ajouté ci-dessus)


def compute_stats(t: np.ndarray, nominal_hz: float) -> dict:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return {}

    nominal_dt = 1.0 / nominal_hz
    inst_rate = np.where(dt > 0, 1.0 / dt, np.inf)

    n = len(t)
    duration = float(t[-1] - t[0])
    mean_rate = (n - 1) / duration if duration > 0 else 0.0

    finite_rate = inst_rate[np.isfinite(inst_rate)]

    probable_gaps = int(np.sum(dt > 1.5 * nominal_dt))
    probable_dupes = int(np.sum(dt <= 0))
    est_missing = float(np.sum(np.maximum(dt / nominal_dt - 1.0, 0.0))) if nominal_dt > 0 else float("nan")

    return {
        "n": n,
        "duration_s": duration,
        "mean_hz": mean_rate,
        "median_hz": float(np.median(finite_rate)) if len(finite_rate) else float("nan"),
        "p1_hz": float(np.percentile(finite_rate, 1)) if len(finite_rate) else float("nan"),
        "p99_hz": float(np.percentile(finite_rate, 99)) if len(finite_rate) else float("nan"),
        "median_dt_ms": float(np.median(dt)) * 1000.0,
        "probable_gaps": probable_gaps,
        "probable_dupes": probable_dupes,
        "est_missing_samples": est_missing,
    }


def print_row(label: str, source: str, stats: dict, nominal_hz: float):
    if not stats:
        print(f"| {label} | (aucune donnée exploitable) |")
        return
    print(f"### {label} (source détectée : {source})\n")
    print(f"- N échantillons/notifications décodés : {stats['n']}")
    print(f"- Durée : {stats['duration_s']:.1f} s")
    print(f"- Cadence moyenne : {stats['mean_hz']:.1f} Hz  (consigne : {nominal_hz:.0f} Hz)")
    print(f"- Cadence médiane (instantanée) : {stats['median_hz']:.1f} Hz")
    print(f"- Période médiane : {stats['median_dt_ms']:.2f} ms")
    print(f"- Centile 1 / centile 99 (cadence instantanée) : {stats['p1_hz']:.1f} Hz / {stats['p99_hz']:.1f} Hz")
    print(f"- Sauts probables (intervalle > 1.5x période nominale) : {stats['probable_gaps']}")
    print(f"- Doublons/désordre probables (dt <= 0) : {stats['probable_dupes']}")
    print(f"- Échantillons manquants estimés (somme des sauts, borne basse) : "
          f"{stats['est_missing_samples']:.0f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="Enregistrement GUI CSV ou export nRF Connect (BLE)")
    ap.add_argument("--label", default=None, help="Étiquette pour la ligne du tableau 4.2")
    ap.add_argument("--nominal-hz", type=float, default=250.0,
                     help="Cadence nominale attendue (consigne firmware, défaut 250 Hz)")
    args = ap.parse_args()

    if not args.log.is_file():
        print(f"Fichier introuvable : {args.log}", file=sys.stderr)
        return 1

    data = load_log(args.log)
    stats = compute_stats(data["t"], args.nominal_hz)
    label = args.label or args.log.stem

    print_row(label, data["source"], stats, args.nominal_hz)

    if data["source"] == "nrf_connect":
        print(
            "Avertissement (BLE) : la cadence ci-dessus est celle des ÉVÉNEMENTS "
            "'value received' vus par nRF Connect, pas nécessairement celle des "
            "échantillons individuels si le firmware groupe plusieurs échantillons "
            "par notification (le rapport mentionne des lots de 20, section 3.3). "
            "Si chaque ligne du log correspond à un échantillon déjà dégroupé par "
            "nRF Connect, ce résultat est directement comparable à la colonne USB. "
            "Sinon, diviser la cadence de notification par la taille de lot pour "
            "obtenir la cadence par échantillon, et documenter cette conversion "
            "dans le rapport plutôt que de la passer sous silence."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
