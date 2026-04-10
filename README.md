# EcoMetrics: Socio-Economic Determinants of CO2 Emissions

## Despre Proiect
**EcoMetrics** este o platformă interactivă de analiză a datelor dezvoltată pentru a explora relația complexă dintre dezvoltarea economică, urbanizare și amprenta de carbon la nivel global. Utilizând un set de date de tip *cross-section* (date regionale), aplicația investighează modul în care indicatori precum PIB/capita, densitatea populației și structura economiei (servicii vs. industrie) dictează nivelul emisiilor de CO2.

## Obiectivul Analizei
Scopul cercetării este de a identifica pattern-uri de sustenabilitate prin:
* **Analiză Comparativă:** Evaluarea discrepanțelor de emisii între economiile emergente și cele dezvoltate.
* **Modelare Predictivă:** Clasificarea statelor în categorii de risc climatic în funcție de profilul lor socio-economic.
* **Segmentare:** Gruparea țărilor în clustere omogene pentru a propune politici de mediu adaptate contextului regional.

## Pipeline Tehnic
1.  **Data Engineering:** Curățarea automată a datelor, tratarea asimetriilor prin logaritmare și standardizarea variabilelor pentru eliminarea erorilor de unitate de măsură.
2.  **Exploratory Data Analysis (EDA):** Identificarea outlierilor și vizualizarea distribuțiilor statistice.
3.  **Analytics:** Integrarea modelelor de Machine Learning (Clasificare Logistică și Clusterizare K-Means) pentru interpretarea cauzalității.

## Pasi de instalare

1. Creare mediu: python -m venv venv
2. Activare: venv\Scripts\activate (Windows) sau source venv/bin/activate (Mac/Linux)
3. Instalare librării: pip install -r requirements.txt
