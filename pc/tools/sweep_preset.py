#!/usr/bin/env python3
"""
tools/sweep_preset.py
──────────────────────
Balaie un seul paramètre matériel (BOOST, GAIN, LED, ou ADSGAIN) pendant que
les autres restent fixes, enregistre le signal brut à chaque valeur, et
trace l'AC/DC (%) et le SNR intra-bande en fonction du paramètre.

Objectif : trouver visuellement le "knee point" — le point où monter le
paramètre commence à ajouter plus de bruit (dark count SiPM, saturation)
que de signal utile — plutôt que de deviner un preset ou de laisser les
deux AGC (AutoTuner + BoostOptimizer) tourner en même temps et se marcher
dessus.

Utilisation
───────────
  # Balayer le BOOST de 20 à 120 par pas de 10, 5 s par valeur
  python tools/sweep_preset.py --port /dev/tty.usbserial-0001 \\
      --param BOOST --range 20:120:10

  # Balayer les LEDs avec des valeurs précises
  python tools/sweep_preset.py --param LED --values 60,90,115,140,180

  # Ajuster la durée d'enregistrement et le temps de stabilisation
  python tools/sweep_preset.py --param GAIN --range 10:128:15 \\
      --duration 6 --settle 1.5

Le script suppose une position/contact du capteur STABLE pendant tout le
balayage (bras immobile posé sur une table) — sinon les artefacts de
mouvement fausseront la comparaison entre les valeurs.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports
from scipy.signal import butter, find_peaks, sosfiltfilt

# Permet de lancer ce script directement, sans configurer PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.agc import ADC_FULL_SCALE
from pipeline.core import _despike, inband_snr, robust_ac_amplitude

SWEEP_DIR = Path(__file__).resolve().parents[2] / "data" / "recordings" / "sweeps"

# Commande(s) série à envoyer pour chaque type de paramètre balayé.
# LED envoie LED1 et LED2 ensemble (ils bougent toujours de concert dans l'app).
PARAM_COMMANDS = {
    "BOOST":   lambda v: [f"BOOST:{v}"],
    "GAIN":    lambda v: [f"GAIN:{v}"],
    "LED":     lambda v: [f"LED1:{v}", f"LED2:{v}"],
    "ADSGAIN": lambda v: [f"ADSGAIN:{v}"],
}

# Valeurs par défaut des paramètres NON balayés (reprises des defaults GUI).
DEFAULTS = {"BOOST": 63, "GAIN": 34, "LED": 115, "ADSGAIN": 0}

# Fraction de la plage ADC considérée comme saturation (près des rails).
SAT_MARGIN = 0.03


@dataclass
class StepResult:
    value: int
    dc: float
    ac: float
    acdc_pct: float
    inband_snr: float
    sat_fraction: float
    n_samples: int
    n_beats: int = 0            # nombre de pics périodiques détectés (pas juste une amplitude)
    settling_ok: bool = True    # False si le DC dérive encore à la fin de la fenêtre
    trustworthy: bool = False   # True si AC/DC + périodicité + stabilisation sont tous OK


def autodetect_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        dev = p.device.lower()
        if any(k in dev for k in ("usbserial", "slab", "wchusb", "cu.usb", "ttyusb", "ttyacm")):
            return p.device
    return ports[0].device


def send(ser: serial.Serial, cmd: str):
    ser.write((cmd + "\n").encode("utf-8"))


def set_baseline(ser: serial.Serial, param: str, base: dict[str, int]):
    """Envoie les valeurs de repos pour tous les paramètres SAUF celui balayé."""
    for name, cmds in PARAM_COMMANDS.items():
        if name == param:
            continue
        for cmd in cmds(base[name]):
            send(ser, cmd)
    time.sleep(0.3)


def record_step(ser: serial.Serial, duration_s: float) -> np.ndarray:
    """Lit le port pendant duration_s et retourne le PPG brut (led1) en array."""
    ser.reset_input_buffer()
    values: list[int] = []
    buf = b""
    t_end = time.time() + duration_s
    while time.time() < t_end:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode("utf-8", "ignore").strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) != 5:
                continue
            try:
                values.append(int(parts[0]))
            except ValueError:
                continue
    return np.asarray(values, dtype=float)


def _count_periodic_beats(x: np.ndarray, fs: float,
                          min_beats: int = 4,
                          max_ibi_cv: float = 0.35) -> tuple[int, float]:
    """
    Compte les pics périodiques plausibles (rythme cardiaque) dans un signal
    déjà centré sur 0. Rejette les transitoires uniques (rampe de
    stabilisation, glitch série) qui n'ont l'air d'un battement qu'au
    premier coup d'œil sur l'amplitude.

    Retourne (nombre_de_pics, coefficient_de_variation_des_intervalles).
    Un vrai battement régulier a un CV bas (< ~0.35). Un artefact isolé
    donne 0 ou 1 pic, ou des intervalles très irréguliers.
    """
    if len(x) < int(2 * fs):
        return 0, np.inf

    nyq = fs / 2.0
    sos = butter(4, [0.7 / nyq, min(3.5 / nyq, 0.99)], btype="bandpass", output="sos")
    try:
        filt = sosfiltfilt(sos, x)
    except ValueError:
        return 0, np.inf

    min_dist = max(int(0.3 * fs), 1)               # ≤ 200 BPM
    prominence = max(np.std(filt) * 0.5, 1e-6)
    peaks, _ = find_peaks(filt, distance=min_dist, prominence=prominence)

    if len(peaks) < 2:
        return len(peaks), np.inf

    ibi = np.diff(peaks) / fs
    ibi_cv = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else np.inf
    return len(peaks), ibi_cv


def _detect_residual_drift(raw: np.ndarray, fs: float,
                           max_drift_sigma: float = 8.0) -> bool:
    """
    Compare le DC du premier tiers et du dernier tiers de la fenêtre.
    Si la moyenne a encore bougé de façon importante entre les deux, le
    point n'a probablement pas fini de se stabiliser électriquement — la
    constante de temps RC du TIA peut être plus longue que le --settle
    utilisé, surtout aux valeurs de gain élevées (τ ≈ R_feedback × C_comp
    augmente avec le gain). Indépendant de l'amplitude AC détectée.

    Retourne True si la stabilisation semble OK (pas de dérive suspecte).
    """
    n = len(raw)
    if n < int(3 * fs):
        return True  # trop court pour juger, on ne bloque pas sur ce critère

    third = n // 3
    first_mean = float(np.mean(raw[:third]))
    last_mean = float(np.mean(raw[-third:]))
    local_std = float(np.std(raw[-third:])) or 1.0

    drift_in_sigmas = abs(last_mean - first_mean) / local_std
    return drift_in_sigmas <= max_drift_sigma


def analyze_step(value: int, raw: np.ndarray, fs: float,
                 min_acdc_pct: float = 0.05,
                 discard_s: float = 0.5,
                 min_beats: int = 4,
                 max_ibi_cv: float = 0.35) -> StepResult:
    """
    min_acdc_pct : seuil plancher d'AC/DC (%) en dessous duquel on considère
                   qu'il n'y a pas de vrai signal cardiaque.
    discard_s    : secondes retirées au DÉBUT de la fenêtre, pour ignorer un
                   transitoire de stabilisation résiduel juste après le
                   changement de commande (observé même avec --settle).
    min_beats    : nombre minimum de pics périodiques requis pour faire
                   confiance à l'amplitude AC — un pic isolé (glitch,
                   rampe) ne suffit plus.
    max_ibi_cv   : variabilité maximale tolérée entre les intervalles de
                   pics pour les considérer "réguliers" (rythme cardiaque
                   plausible plutôt que bruit aléatoire).
    """
    n_discard = int(discard_s * fs)
    if n_discard > 0 and len(raw) > n_discard:
        raw = raw[n_discard:]

    if len(raw) < int(2 * fs):
        return StepResult(value, 0.0, 0.0, 0.0, 0.0, 1.0, len(raw), 0,
                          settling_ok=False, trustworthy=False)

    settling_ok = _detect_residual_drift(raw, fs)

    # Retire les glitches ponctuels (échantillon isolé très hors norme)
    # avant tout calcul — sinon un seul point aberrant fausse le percentile.
    clean = _despike(raw, sigma_limit=5.0)

    dc = float(np.median(clean))
    ac = robust_ac_amplitude(clean - dc, fs)
    acdc_pct = 100.0 * ac / abs(dc) if abs(dc) > 1e-9 else 0.0
    snr = inband_snr(clean - dc, fs)

    lo_thr = SAT_MARGIN * ADC_FULL_SCALE
    hi_thr = (1 - SAT_MARGIN) * ADC_FULL_SCALE
    sat_fraction = float(np.mean((raw < lo_thr) | (raw > hi_thr)))

    n_beats, ibi_cv = _count_periodic_beats(clean - dc, fs, min_beats, max_ibi_cv)

    trustworthy = (
        acdc_pct >= min_acdc_pct
        and n_beats >= min_beats
        and ibi_cv <= max_ibi_cv
        and settling_ok
    )

    return StepResult(value, dc, ac, acdc_pct, snr, sat_fraction, len(raw),
                      n_beats, settling_ok, trustworthy)


def parse_values(args) -> list[int]:
    if args.values:
        return [int(v.strip()) for v in args.values.split(",")]
    start, stop, step = (int(x) for x in args.range.split(":"))
    return list(range(start, stop + 1, step))


def print_table(results: list[StepResult], param: str):
    print(f"\n{'='*92}")
    print(f"  Balayage : {param}")
    print(f"{'='*92}")
    print(f"{'Valeur':>8} {'DC':>8} {'AC':>8} {'AC/DC %':>9} "
          f"{'SNR':>8} {'Battements':>10} {'Saturation':>11}  {'Statut'}")
    print("-" * 92)
    for r in results:
        if r.sat_fraction > 0.02:
            status = "⚠ SATURÉ"
        elif not r.settling_ok:
            status = "⏱ pas stabilisé (augmente --settle)"
        elif not r.trustworthy:
            status = "✗ pas de signal fiable (bruit/artefact)"
        else:
            status = "✓ signal réel périodique"
        print(f"{r.value:>8} {r.dc:>8.0f} {r.ac:>8.2f} {r.acdc_pct:>8.3f}% "
              f"{r.inband_snr:>8.2f} {r.n_beats:>10} {100*r.sat_fraction:>10.1f}%  {status}")


def pick_recommendation(results: list[StepResult]) -> StepResult | None:
    """
    Meilleur SNR parmi les points qui ont à la fois :
      - un vrai signal cardiaque (trustworthy = AC/DC au-dessus du plancher de bruit)
      - pas de saturation
    Sans ce double filtre, le SNR calculé sur du bruit plat peut sembler
    élevé par hasard statistique et fausser complètement la recommandation.
    """
    usable = [r for r in results
              if r.sat_fraction <= 0.02 and r.trustworthy and r.n_samples > 0]
    if not usable:
        return None
    return max(usable, key=lambda r: r.inband_snr)


def plot_results(results: list[StepResult], param: str, save_path: Path):
    import matplotlib.pyplot as plt

    values = [r.value for r in results]
    acdc = [r.acdc_pct for r in results]
    snr = [r.inband_snr for r in results]
    sat = [r.sat_fraction > 0.02 for r in results]
    flat = [not r.trustworthy for r in results]

    def color_for(is_sat: bool, is_flat: bool, base: str) -> str:
        if is_sat:
            return "#c0392b"      # rouge — saturé
        if is_flat:
            return "#95a5a6"      # gris — pas de signal réel (bruit plat)
        return base               # couleur normale — signal réel exploitable

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f"Balayage {param} — recherche du knee point", fontsize=13)

    colors1 = [color_for(s, f, "#27ae60") for s, f in zip(sat, flat)]
    ax1.bar(values, acdc, width=(values[1] - values[0]) * 0.6 if len(values) > 1 else 5,
            color=colors1)
    ax1.set_ylabel("AC/DC (%)")
    ax1.set_title("Amplitude du signal cardiaque relative au DC")
    ax1.grid(True, alpha=0.3)

    colors2 = [color_for(s, f, "#2980b9") for s, f in zip(sat, flat)]
    ax2.bar(values, snr, width=(values[1] - values[0]) * 0.6 if len(values) > 1 else 5,
            color=colors2)
    ax2.set_ylabel("SNR intra-bande (fiable seulement si signal réel)")
    ax2.set_xlabel(param)
    ax2.set_title("Qualité du signal — rouge=saturé, gris=bruit plat (SNR non fiable)")
    ax2.grid(True, alpha=0.3)

    rec = pick_recommendation(results)
    if rec:
        ax2.axvline(rec.value, color="#f39c12", linestyle="--", linewidth=1.5,
                    label=f"recommandé: {param}={rec.value}")
        ax1.axvline(rec.value, color="#f39c12", linestyle="--", linewidth=1.5)
        ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nGraphique sauvegardé : {save_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="Port série (auto-détecté si omis)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--param", required=True, choices=list(PARAM_COMMANDS.keys()),
                     help="Paramètre à balayer")
    ap.add_argument("--values", default=None,
                     help="Liste explicite, ex: 20,40,63,80,100")
    ap.add_argument("--range", default=None,
                     help="start:stop:step, ex: 20:120:10")
    ap.add_argument("--duration", type=float, default=5.0,
                     help="Secondes enregistrées par valeur (défaut: 5)")
    ap.add_argument("--settle", type=float, default=1.0,
                     help="Secondes d'attente après le changement avant d'enregistrer (défaut: 1)")
    ap.add_argument("--fs", type=float, default=250.0,
                     help="Fréquence d'échantillonnage nominale du firmware (défaut: 250 Hz)")
    ap.add_argument("--base-boost", type=int, default=DEFAULTS["BOOST"])
    ap.add_argument("--base-gain", type=int, default=DEFAULTS["GAIN"])
    ap.add_argument("--base-led", type=int, default=DEFAULTS["LED"])
    ap.add_argument("--base-adsgain", type=int, default=DEFAULTS["ADSGAIN"])
    ap.add_argument("--min-acdc", type=float, default=0.05,
                     help="Seuil plancher AC/DC %% en dessous duquel un point est "
                          "considéré comme bruit plat, pas un vrai signal (défaut: 0.05)")
    ap.add_argument("--discard", type=float, default=0.5,
                     help="Secondes retirées au début de chaque fenêtre pour ignorer "
                          "un transitoire résiduel de stabilisation (défaut: 0.5)")
    ap.add_argument("--min-beats", type=int, default=4,
                     help="Nombre minimum de pics périodiques requis pour valider un "
                          "point comme signal cardiaque réel, pas un artefact isolé "
                          "(défaut: 4)")
    ap.add_argument("--save-raw", action="store_true",
                     help="Sauvegarde le signal brut de chaque étape en CSV pour "
                          "inspection visuelle ultérieure (tools/visualize_log.py)")
    args = ap.parse_args()

    if not args.values and not args.range:
        ap.error("Spécifie --values ou --range")

    base = {
        "BOOST": args.base_boost,
        "GAIN": args.base_gain,
        "LED": args.base_led,
        "ADSGAIN": args.base_adsgain,
    }
    values = parse_values(args)

    port = args.port or autodetect_port()
    if not port:
        print("Aucun port série trouvé. Branche l'ESP32 ou spécifie --port.")
        return 1

    print(f"# Port: {port} @ {args.baud} baud")
    print(f"# Paramètre balayé: {args.param}  →  valeurs: {values}")
    print(f"# Valeurs fixes: { {k: v for k, v in base.items() if k != args.param} }")
    print(f"# {args.duration}s d'enregistrement + {args.settle}s de stabilisation par valeur")
    print(f"# Durée totale estimée: {len(values) * (args.duration + args.settle):.0f}s\n")

    ser = serial.Serial(port, args.baud, timeout=0.2)
    time.sleep(0.3)

    set_baseline(ser, args.param, base)
    ser.write(b"HVEN:1\n")
    time.sleep(0.3)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = SWEEP_DIR / f"sweep_{args.param.lower()}_{stamp}_raw"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[StepResult] = []
    for v in values:
        for cmd in PARAM_COMMANDS[args.param](v):
            send(ser, cmd)
        print(f"  → {args.param}={v} : stabilisation ({args.settle}s)...", end="", flush=True)
        time.sleep(args.settle)

        print(f" enregistrement ({args.duration}s)...", end="", flush=True)
        raw = record_step(ser, args.duration)
        result = analyze_step(v, raw, args.fs, min_acdc_pct=args.min_acdc,
                              discard_s=args.discard, min_beats=args.min_beats)
        results.append(result)

        if args.save_raw and len(raw) > 0:
            np.savetxt(raw_dir / f"{args.param.lower()}_{v}.csv", raw,
                      delimiter=",", header="led1_adc", comments="")

        status = "✓" if result.trustworthy and result.sat_fraction <= 0.02 else "✗"
        print(f" AC/DC={result.acdc_pct:.3f}%  SNR={result.inband_snr:.2f}"
              f"  sat={100*result.sat_fraction:.1f}%  {status}")

    ser.close()

    print_table(results, args.param)

    rec = pick_recommendation(results)
    if rec:
        print(f"\n>>> Recommandation : {args.param} = {rec.value} "
              f"(SNR={rec.inband_snr:.2f}, AC/DC={rec.acdc_pct:.3f}%, sans saturation)")
    else:
        print("\n>>> Aucun point sans saturation — élargis la plage ou baisse les valeurs.")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = SWEEP_DIR / f"sweep_{args.param.lower()}_{stamp}.png"
    plot_results(results, args.param, plot_path)
    if args.save_raw:
        print(f"Signaux bruts sauvegardés : {raw_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())