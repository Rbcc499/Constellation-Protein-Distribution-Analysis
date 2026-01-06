<img width="1223" height="706" alt="image" src="https://github.com/user-attachments/assets/7069a8b5-aad6-4b85-b26b-bb89f320d423" />

**A Constellation of Connections**
- Map synaptic protein data with Constellation. 

- Constellation is a desktop GUI for running proteome & protein-distance analyses from a folder of exported analysis files plus an expansion factor spreadsheet. It provides a clean, fullscreen interface with optional Alignment, Angle, and Enrichment analyses.

**Created by: Rebecca E. Twilley**

## Features

- **Data Inputs**
  - Select a **Data Folder** (your exported analysis file from Imaris (`.xlsx, .csv`), find template excel file in template folder)
  - Select an **Expansion Factor** Excel file (`.xlsx, .csv`), find template excel file in template folder)

- **Analyses (Toggle On/Off)**
  - **Alignment analysis**
    - Set **protein locations** (presynaptic/postsynaptic; membrane/cytosol)
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

## Steps for Use 
  **1)**  Upload data folder containing Imaris derived excel files only.* Headers should be as per supplied template. Constellation will load excel files and read protein names. <br />
  **2)**  Upload expansion factor excel Headers should be a per supplied expansion_factor_template file. <br />
  **3)**  Select the parameters you require for your analysis. <br />
    - For Alignment Analysis: Select the location of each protein, then select which protein pair that will have the longest distance. <br />
    - For Angle Analysis: Assign which proteins will be included in the angle analysis. Then put the angle threshold (degrees). <br />
  **4)**  Click run analysis. Results will be saved in the data folder you uploaded. <br />
  **5)**  Click refresh to refresh program to run another analysis. <br />

  <br />
<img width="1919" height="1077" alt="constellation_example_layout" src="https://github.com/user-attachments/assets/ff1916be-9c4d-4326-8135-4cef212d4e63" />

<br />

*This is an expansion on the original code obtained from: https://github.com/SiddiquiLab/Padmanabhan-et-al.-/blob/main/README.md Imaris and Huygens workflows can be found there.
