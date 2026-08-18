#!/usr/bin/env python3
"""Convertit un fichier .fit Garmin (activité enregistrée sur la montre) en
CSV de référence au format attendu par tools/validate_rest_hr.py :

    unix_time_s,bpm_ref

Protocole recommandé pour une session de validation
-----------------------------------------------------
1. Démarrer une activité sur la montre (ex. "Marche" ou un simple chrono
   cardio) juste avant de cliquer "Start CSV recording" dans la GUI, pour
   que les deux enregistrements se chevauchent dans le temps.
2. À la fin, arrêter l'activité sur la montre et le recording GUI.
3. Synchroniser la montre (Garmin Connect / Garmin Express) puis
   télécharger le fichier .fit original de l'activité (sur Garmin Connect :
   page de l'activité -> engrenage -> "Exporter le fichier d'origine").
4. Convertir :

       python tools/fit_to_reference.py activity.fit -o reference.csv

5. Alimenter validate_rest_hr.py :

       python tools/validate_rest_hr.py --pair recording.csv reference.csv \
           --label "Sujet 1" --plot rapport/fig_4_1_sujet1.png

Note importante sur l'horodatage : les timestamps du .fit sont en UTC absolu
(horloge de la montre), tout comme unix_time_s dans l'enregistrement GUI
(time.time()). Aucune conversion de fuseau horaire n'est nécessaire tant que
l'horloge du PC et celle de la montre sont correctement synchronisées (NTP /
sync automatique). Si les deux enregistrements ne se recouvrent pas du tout
une fois convertis, vérifier en premier l'horloge de la montre.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fitparse import FitFile
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Le module 'fitparse' est requis pour lire les fichiers .fit Garmin.\n"
        "Installer avec : pip install fitparse"
    ) from exc


def extract_hr_records(fit_path: Path) -> list[tuple[float, int]]:
    """Extrait (unix_time_s, bpm) de chaque message 'record' du .fit qui
    contient une valeur de fréquence cardiaque."""
    fitfile = FitFile(str(fit_path))
    out: list[tuple[float, int]] = []
    for record in fitfile.get_messages("record"):
        ts = record.get_value("timestamp")
        hr = record.get_value("heart_rate")
        if ts is None or hr is None:
            continue
        unix_s = ts.replace(tzinfo=timezone.utc).timestamp()
        out.append((unix_s, int(hr)))
    out.sort(key=lambda p: p[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit_file", type=Path, help="Fichier .fit exporté de la montre (activité enregistrée)")
    ap.add_argument("-o", "--output", type=Path, required=True,
                     help="CSV de sortie (format : unix_time_s,bpm_ref)")
    args = ap.parse_args()

    if not args.fit_file.is_file():
        print(f"Fichier introuvable : {args.fit_file}", file=sys.stderr)
        return 1

    records = extract_hr_records(args.fit_file)
    if not records:
        print(
            f"Aucun échantillon de fréquence cardiaque trouvé dans {args.fit_file}.\n"
            "Vérifier qu'il s'agit bien d'un fichier d'ACTIVITÉ (pas d'un fichier de "
            "monitoring quotidien) et que le capteur cardiaque de la montre était actif "
            "pendant l'enregistrement.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unix_time_s", "bpm_ref"])
        writer.writerows(records)

    t0, t1 = records[0][0], records[-1][0]
    print(
        f"{len(records)} échantillons HR extraits de {args.fit_file.name} "
        f"({(t1 - t0) / 60.0:.1f} min, unix_time_s {t0:.0f} -> {t1:.0f})\n"
        f"Écrit : {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())