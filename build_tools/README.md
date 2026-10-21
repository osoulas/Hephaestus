# Conda environment

From the repository root, create and activate the optional Conda environment:

```bash
conda env create -f build_tools/environment.yml
conda activate hephaestus
python -m pip install -e ".[sim,dev]"
```

The project metadata and dependencies stay in the root `pyproject.toml`.
