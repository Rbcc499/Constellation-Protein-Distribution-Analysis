# Constellation (Proteome Analysis GUI)

Constellation is a desktop GUI for running proteome / protein-distance analyses from a folder of exported analysis files plus an expansion factor spreadsheet. It provides a clean, fullscreen interface with optional Alignment, Angle, and Enrichment analyses.

> Created by: Rebecca E. Twilley

---

## Features

- **Data Inputs**
  - Select a **Data Folder** (your exported analysis files)
  - Select an **Expansion Factor** Excel file (`.xlsx`)

- **Analyses**
  - **Alignment analysis**
    - Choose the *Longest Protein Pair* from a dropdown
    - Set **protein locations** (presynaptic/postsynaptic; membrane/cytosol)
    - Set an **alignment tolerance**
  - **Angle analysis**
    - Pick **3 proteins (A, B, C)** to form a triangle
    - Set max angle thresholds at A/B/C (degrees)
  - **Enrichment analysis**
    - Set an enrichment cutoff (µm)

- **Parameters**
  - Number of proteins: **3 or 4**
  - Cluster distance cutoff (µm) (e.g., **0.3 µm = 300 nm**)

- **UX**
  - Fullscreen by default
  - Press **Esc** to exit fullscreen, **F11** to toggle fullscreen

This is an expansion on the original code obtained from: https://github.com/SiddiquiLab/Padmanabhan-et-al.-/blob/main/README.md
