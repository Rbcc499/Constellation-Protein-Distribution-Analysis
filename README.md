<img width="1223" height="706" alt="image" src="https://github.com/user-attachments/assets/7069a8b5-aad6-4b85-b26b-bb89f320d423" />

## A Constellation of Connections

- Constellation is a desktop GUI for running proteome & protein-distance analyses from a folder of exported analysis files plus an expansion factor spreadsheet. It provides a clean, fullscreen interface with optional Alignment, Angle, and Enrichment analyses. <br />

**Created by: Rebecca E. Twilley** <br /><br />

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

**File Nomenclature**: Files received from the imaris output need to have a specific nomenclature for the program to read. The columns inside the excel file also need specific names. For template excel files please see the template folder. <br />
**mouseid_sliceid_roundnumber. It is imperative that the mouse id, sliceid, and roundnumber are in this order, with '_' separating. You may add additional information afterwards separated by a '_'.** <br />
**For example:** M123L_s1h1_1 (mouseid = M123L, sliceid = s1h1, round_number = 1)
**Correct Example:** M123L_s1h1_1_PFA_jan_1_2000, this is a correct way of naming the file
**Incorrect Example:** M123L_jan_1_200_s1h1_1, this is incorrect, the file should start with the mouseid, followed by the sliceid, followed by the roundnumber, then extra information may follow) 

  **1)**  Upload data folder containing Imaris derived excel files only.* Headers should be as per supplied template. Constellation will load excel files and read protein names. <br />
  **2)**  Upload expansion factor excel Headers should be a per supplied expansion_factor_template file. <br />
  **3)**  Select the parameters you require for your analysis. <br />
    - For Alignment Analysis: Select the location of each protein, then select which protein pair that will have the longest distance. <br />
    - For Angle Analysis: Assign which proteins will be included in the angle analysis. Then put the angle threshold (degrees). <br />
  **4)**  Click run analysis. Results will be saved in the data folder you uploaded. <br />
  **5)**  Click refresh to refresh program to run another analysis. <br />

  <br />

![constellation_example_setup](https://github.com/user-attachments/assets/ed1604d1-fb93-48fc-bc55-da89b43410b8)

<br />

*This is an expansion on the original code obtained from: https://github.com/SiddiquiLab/Padmanabhan-et-al.-/blob/main/README.md Imaris and Huygens workflows can be found there.
