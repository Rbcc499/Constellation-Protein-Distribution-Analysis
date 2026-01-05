<img width="1223" height="706" alt="image" src="https://github.com/user-attachments/assets/7069a8b5-aad6-4b85-b26b-bb89f320d423" />

**A Constellation of Connections**

Map synaptic protein data with Constellation. Constellation is a desktop GUI for running proteome & protein-distance analyses from a folder of exported analysis files plus an expansion factor spreadsheet. It provides a clean, fullscreen interface with optional Alignment, Angle, and Enrichment analyses.

**Created by: Rebecca E. Twilley**
Map  — a desktop GUI for protein-distance and proteome analyses. Load a folder of exported analysis files plus an expansion factor spreadsheet, then run a clean, fullscreen workflow with optional Alignment, Angle, and Enrichment analyses.
## Features

- **Data Inputs**
  - Select a **Data Folder** (your exported analysis file from Imaris, find template excel file in template folder)
  - Select an **Expansion Factor** Excel file (`.xlsx`, find template excel file in template folder)

- **Analyses (Toggle On/Off)**
  - **Alignment analysis**
    - First set **protein locations** (presynaptic/postsynaptic; membrane/cytosol)
    - Choose the *Longest Protein Pair* from a dropdown (formulated after protein locations are set)
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
