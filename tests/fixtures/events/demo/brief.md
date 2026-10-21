# Experimental brief: SrTi0.75Fe0.25O3

## Recommended composition and structure

**SrTi0.75Fe0.25O3** (c2-1), space group Pm-3m, structure from template:perovskite_cubic. Picked by orchestrator.

## Predicted properties against target

| Property | Predicted | Target | Met |
| --- | --- | --- | --- |
| band_gap (eV) | 2.2 ± 0.65 | < 3.12 | yes (conservative: yes) |
| Energy above hull (eV/atom) | 0.02 | ≤ 0.05 | yes |
| Dynamic stability (min frequency, THz) | 0.2 | mode mapping | yes |

## Physical rationale

Fe 3d states narrow the gap (stub:fe-sto).

## Validation verdict

- c2-1: **confirmed**.
  - e_hull_eV_per_atom: 0.02 → 0.024
  - stub final check

## Synthesis route

- Solid state from SrCO3 + TiO2 at 1200 C. [stub:sto-ss]

## Characterisation plan

1. XRD: confirm the perovskite phase and rule out secondary phases.
2. UV-vis diffuse reflectance (Tauc plot): measure the optical gap.
3. Raman: check for octahedral tilts the cubic model may miss.
4. XPS: confirm dopant oxidation states.

## Risks and fallbacks

- (major) MEGNet ±0.65 eV.
- (minor) Route known.
- Fallback: the runner-up candidate below.

## Runner-up candidates

- SrTi0.875Fe0.125O3 (c1-1): band_gap 2.6 ± 0.65 eV, E_hull 0.03 eV/atom, dynamically stable: yes.

## Iteration history

- Round 1: SrTi0.875Fe0.125O3, BaTi0.875Fe0.125O3; best c1-1, rubric 75, reviewer confidence 60.
- Round 2: SrTi0.75Fe0.25O3; best c2-1, rubric 100, reviewer confidence 80.

## Literature support

- [stub:fe-sto] Fe-doped SrTiO3: visible-light absorption from Fe 3d states

## Scope and assumptions

- Operating conditions default to 300 K and ambient pressure (GP-7).
- Target value is the model's own prediction for SrTiO3 (3.12 eV), so candidates and reference sit on the same scale (GP-11).
- Screening potential mace-mp-0-small; final re-check mace-mp-0-medium.
- Band gap model at HSE fidelity; values are on the model's scale.
- Literature comes from Valency Bond, which indexes arXiv, bioRxiv, medRxiv and PubMed records; journal-only papers with no preprint are missing (LM-19).
- Supported structures: cubic perovskite template and Materials Project polymorphs.
