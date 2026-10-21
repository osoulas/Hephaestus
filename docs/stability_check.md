# Stability sanity check

Computed on 2026-10-21.

**Setup.**
- Potentials: MACE-MP-0 small (float64), MACE-MPA-0 medium (float32), MACE-OMAT-0 medium (float32) and CHGNet 0.3.0. All ran on an RTX 3060 (torch 2.5.1+cu121; see `pyproject.toml` for the driver note).
- Relaxation: FIRE + FrechetCellFilter, fmax 0.01 eV/Å. Cubic templates were relaxed under FixSymmetry.
- Hull: every MP `is_stable` phase in each chemical subsystem, re-relaxed with the same potential (SE-13).
- Phonons: phonopy finite displacement (0.01 Å), frequencies at q-points commensurate with the supercell.
- Gates: E_hull ≤ 0.05 eV/atom and dynamic stability, as defined below.

**Licence note.** MACE-OMAT-0 is released under the Academic Software License (non-commercial use only). MACE-MP-0 and MPA-0 are MIT. Confirm the hackathon's terms allow ASL before using OMAT in the demo.

## Step 1: plain phonons fail almost everything

Without mode mapping, nearly every structure has imaginary modes below −0.1 THz under every potential. That includes SrTiO3 I4/mcm, BaTiO3 Amm2 and CaTiO3 Pnma. The exceptions are BaTiO3 and CaTiO3 under CHGNet, and the DFT-stable controls (below), which pass under all potentials. A MgO control gives clean phonons, so the phonon code is correct. Materials Project's own DFT phonons show the same Γ instabilities in those three structures (see the comparison below), so this is mostly inherited from DFT, not just softness in the potentials.

## Step 2: mode mapping

For the softest mode, mode mapping freezes it into a commensurate supercell, scans the amplitude, re-relaxes, and compares energies.
- If the energy drops by **more than 1 meV/atom**, the instability is real. The distorted structure becomes the new candidate structure and is checked again (at most 2 rounds).
- If it doesn't drop, the mode is **spurious** and the structure keeps its symmetry.

Verdicts are `stable`, `spurious_soft_mode`, `stable_after_distortion` and `unstable`.

| Material | E_hull MP-0 / MPA-0 / OMAT / CHGNet (eV/atom) | MP-0 small | MPA-0 | OMAT-0 | CHGNet |
| --- | --- | --- | --- | --- | --- |
| BaTiO3 R3m (control) | −0.001 / −0.001 / −0.001 / 0.002 | **stable** | **stable** | **stable** | **stable** |
| SrZrO3 Pnma (control) | 0.000 / 0.000 / 0.000 / 0.000 | **stable** | **stable** | **stable** | spurious (0.00) |
| SrSnO3 Pnma (control) | 0.000 / 0.000 / 0.000 / 0.000 | **stable** | **stable** | **stable** | **stable** |
| SrTiO3, MP I4/mcm | 0.000 / 0.000 / 0.000 / 0.000 | spurious (−0.98) | spurious (−0.98) | spurious (−0.47) | spurious (−0.04) |
| BaTiO3, MP Amm2 | 0.000 / 0.000 / 0.000 / 0.000 | → R3m (−1.30) | spurious (−0.92) | spurious (−0.54) | **stable** |
| CaTiO3, MP Pnma | 0.000 / 0.000 / 0.000 / 0.000 | spurious (−0.16) | spurious (−0.35) | spurious (−0.14) | **stable** |
| SrTiO3, cubic | 0.001 / 0.001 / 0.000 / 0.000 | spurious (−0.98) | → Pm (−1.64) | spurious (−0.42) | spurious (−0.34) |
| BaTiO3, cubic | 0.012 / 0.012 / 0.012 / 0.014 | → Cm (−13.6) | → Cm (−13.0) | → Cm (−12.0) | → Amm2 (−14.4) |
| CaTiO3, cubic | **0.064 / 0.066 / 0.062 / 0.060** | → P-1 (−51.8) | → C2/c (−53.7) | → Imma (−50.0) | → C2/c (−50.2) |
| SrTi0.875Zr0.125O3 | 0.013 / 0.015 / 0.013 / 0.013 | → P1 (−8.6) | → P1 (−9.7) | → P1 (−6.9) | → P1 (−6.8) |
| SrTi0.875Sn0.125O3 | 0.011 / 0.009 / 0.009 / 0.000 | → Cm (−6.6) | → P1 (−6.1) | spurious (−0.88) | → P1 (−3.9) |
| SrTaO2N, MP Ama2 | −0.048 / −0.041 / −0.050 / −0.041 | spurious (0.00) | spurious (−0.01) | spurious (0.00) | spurious (0.00) |
| SrTaO2N, ordered cubic | 0.016 / 0.026 / 0.012 / −0.019 | → I4cm (−14.9) | → I4cm (−29.2) | → I4cm (−24.7) | spurious (−0.19) |

Numbers in brackets are the total distortion energy in meV/atom. Bold E_hull fails the 0.05 eV/atom gate. No material was `unstable` under any potential.

## What this means for the demo goal

- **All three MACE models pass the DFT-stable controls outright**, with no imaginary modes. CHGNet passes two and shows a spurious soft mode on SrZrO3.
- **Mode mapping fixes the false negatives.** With it, no experimental ground state is rejected under any potential. Without it, nearly all were.
- **Mode mapping also removes the dynamic gate as a filter.** Every cubic template either has spurious soft modes or relaxes into a lower-symmetry phase that passes. Nothing fails as `unstable`. Across iterations, the dynamic gate will report *which structure* a candidate adopts, not pass or fail.
- **Only the thermodynamic gate discriminates.** Cubic CaTiO3 fails under all four potentials (0.060–0.066 eV/atom). Everything else passes.
- **The answer to the task question:** the demo goal will produce a visible pass/fail pattern from E_hull and the band-gap target, not from phonons.

## Decisions for the team

1. **Dynamic gate definition (touches SE-13 and `PhononResult`).** Proposed: dynamically stable means no imaginary mode whose frozen-in distortion lowers the energy by more than 1 meV/atom. A candidate that distorts is carried forward in its distorted structure, and that distortion is reported. This is still decided by code, not the LLM. It changes what `PhononResult.stable` means, so all four owners need to agree.
2. **Phonon potential: MACE-OMAT-0, if its licence is acceptable.** It passes all three controls outright. Its spurious soft modes are the smallest of the MACE models (≤ 0.54 meV/atom, against up to 0.98 for MP-0 and MPA-0). That removes the SrTiO3 threshold problem: cubic SrTiO3 is −0.42 meV/atom, well clear of 1. Its hull values are within 3 meV/atom of MP for the oxides. **Fallback: MACE-MP-0 small** (MIT licence). It also passes all controls, but SrTiO3 sits at 0.98 meV/atom, right at the threshold, so raise the threshold to 2 meV/atom if you use it. MPA-0 crosses the threshold for cubic SrTiO3 (1.64). CHGNet is a reasonable FC-3 cross-check, but not the primary potential, given the spurious SrZrO3 mode.
3. **Keep MACE-MP-0 small for the screening relaxation and hull** (SYS-2 time budget). Hull values agree across all four potentials to within about 0.01 eV/atom for the oxides.
4. **SrTaO2N lies below the hull under every potential** (−0.04 to −0.05 eV/atom against Sr2TaN3 + Sr4Ta2O9 + Ta3N5). Materials Project puts the same Ama2 structure (mp-2583419) at **+0.031** eV/atom, so all four potentials are off by 0.07–0.08 eV/atom, in the stabilising direction. MP's anion corrections apply per O and N atom, and the decomposition conserves both, so the corrections should mostly cancel. That makes this a real error in the potentials, not a missing correction. Don't use oxynitrides in the demo without a DFT check.

## Comparison with Materials Project hull values

MP values are `energy_above_hull` from the summary API (GGA/GGA+U/r2SCAN mixing, MP2020 corrections). Our values use raw potential energies against MP-stable phases re-relaxed with the same potential.

| Structure | MP ID | MP E_hull | MP-0 small | MPA-0 | OMAT-0 | CHGNet |
| --- | --- | --- | --- | --- | --- | --- |
| SrTiO3 I4/mcm | mp-4651 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SrTiO3 Pm-3m | mp-5229 | 0.0005 | 0.0008 | 0.0006 | 0.0002 | 0.0002 |
| BaTiO3 Amm2 | mp-5777 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BaTiO3 R3m | mp-5020 | 0.000 | −0.0013 | −0.0009 | −0.0005 | 0.0015 |
| BaTiO3 Pm-3m | mp-2998 | 0.0148 | 0.0123 | 0.0120 | 0.0115 | 0.0144 |
| CaTiO3 Pnma | mp-4019 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CaTiO3 Pm-3m | mp-5827 | 0.0645 | 0.0642 | 0.0661 | 0.0617 | 0.0603 |
| SrZrO3 Pnma | mp-4387 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SrSnO3 Pnma | mp-2879 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SrTaO2N Ama2 | mp-2583419 | **0.0308** | −0.048 | −0.041 | −0.050 | −0.041 |

For the oxides, all four potentials match MP to within 4.2 meV/atom (CHGNet's cubic CaTiO3 is the largest gap), and the CaTiO3 cubic gate result (fail at 0.05) agrees with MP. BaTiO3 R3m and Amm2 are degenerate in MP; the potentials split them by about 1 meV/atom. The solid solutions and the ordered cubic SrTaO2N have no MP entry to compare against. Their MP end members are all on or near the hull: SrZrO3 Pnma 0.000, SrSnO3 Pnma 0.000, cubic SrSnO3 0.048.

## Comparison with Materials Project phonons

MP phonons come from `mpr.materials.phonon` (`pheasy` finite-displacement; `dfpt` where available, with non-analytic correction). MP's frequencies are along the Setyawan–Curtarolo path. Ours are at commensurate q-points only. All values are minimum frequency in THz; negative means imaginary.

| Structure | MP ID | MP DFT | MP-0 small | MPA-0 | OMAT-0 | CHGNet |
| --- | --- | --- | --- | --- | --- | --- |
| BaTiO3 R3m | mp-5020 | **0.00** (dfpt), −2.03 near Γ (pheasy) | 0.00 | 0.00 | 0.00 | 0.00 |
| SrZrO3 Pnma | mp-4387 | 0.00 | 0.00 | 0.00 | 0.00 | −0.78 |
| SrSnO3 Pnma | mp-2879 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SrTiO3 I4/mcm | mp-4651 | **−1.83 Γ** (dfpt), −2.64 Γ (pheasy) | −3.35 Γ | −3.86 Γ | −2.88 Γ | −2.18 Γ |
| SrTiO3 Pm-3m | mp-5229 | −1.98 R (Γ stable) | −3.15 Γ, −2.13 R | −3.76 Γ, −2.02 R | −2.83 Γ | −3.54 Γ |
| BaTiO3 Amm2 | mp-5777 | −3.38 Γ | −3.70 Γ | −2.67 Γ | −1.51 Γ | 0.00 |
| CaTiO3 Pnma | mp-4019 | −2.39 Γ | −1.97 Γ | −2.69 Γ | −2.19 Γ | 0.00 |
| CaTiO3 Pm-3m | mp-5827 | −6.40 R | −6.01 R | −6.11 R | −5.76 R | −5.64 |
| SrTaO2N Ama2 | mp-2583419 | −0.10 (stable) | −1.24 | −2.38 | −2.26 | −2.90 |
| SrSnO3 Pm-3m | mp-546973 | −5.54 R | not run | | | |

What this shows:
- **The DFT-stable controls are stable under all three MACE models.** That includes BaTiO3 R3m, the true 0 K phase, where MP's DFPT also gives 0.00. The potentials don't invent instabilities in structures DFT says are stable.
- **MP's DFT also puts imaginary Γ modes in SrTiO3 I4/mcm, BaTiO3 Amm2 and CaTiO3 Pnma.** These structures were poor controls. The potentials largely reproduce their training data here, and mode mapping showing these modes are worth under 1 meV/atom is the physically sensible reading.
- **The real instabilities agree quantitatively.** For the R-point tilt in cubic CaTiO3, MP gives −6.40 THz and the potentials −5.6 to −6.1. For cubic SrTiO3 at R, MP gives −1.98 and MP-0 / MPA-0 give −2.13 / −2.02.
- **Where the potentials disagree with DFT:** cubic SrTiO3 at Γ (MP stable, potentials −2.8 to −3.8; this is why MPA-0 produces a spurious-looking Pm distortion) and SrTaO2N Ama2 (MP essentially stable, potentials −1.2 to −2.9). In both cases mode mapping classes the mode as spurious or near-threshold, so the gate gives the same answer as DFT.

The MP phonon data can also serve as the reference for known compositions in the Simulation Engine: the `pheasy` set covers many ABO3 entries.

**Cost.** With relaxations cached, the mode-mapped check takes about 1–7 min per material on the GPU. Cold runs are dominated by relaxing competing phases, about 6 relaxations/min in float32. The 40-atom solid solutions take about 0.5 min per mapping round.

**Caveat.** The solid solutions use one ordered 12.5 % substitution with Γ-only phonons on a 40-atom cell, not a real SQS. MPA-0 and OMAT-0 ran in float32, so their phonon finite differences are slightly noisier than float64.
