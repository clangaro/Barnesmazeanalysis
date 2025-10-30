# Barnes Maze SLEAP Analysis

A production-ready pipeline for processing SLEAP exports for Barnes Maze:
1) add metadata (file index, mouse id, trial, sex) parsed from filenames  
2) pad CSVs to full frame count using the paired `.analysis.h5`  
3) impute nose coordinates when missing inside the hole (with hysteresis)  
4) produce per-trial summaries and roll-ups  
5) optional plots with SEM and individual points

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
