# genai-tax-thesis-code
Python scripts used for statistical analyses in my Master’s Thesis, Tilburg University.

Python Scripts – Master’s Thesis

This repository contains the Python scripts used for data preparation and statistical analysis in my Master’s Thesis on the adoption of Generative AI in corporate tax functions.

📂 Repository Contents

preprocess_to_scores.py
* Purpose: Preprocesses the raw survey data.
* Main tasks:
- Cleaning survey responses (handling missing values, recoding Likert scales).
- Calculating construct scores (e.g., technological readiness, organizational readiness, environmental readiness, trust, adoption).
- Exporting a cleaned dataset ready for analysis.

Appendix_D_analysis.py
* Purpose: Conducts all statistical analyses used in the thesis.
* Main tasks:
- Reliability tests (Cronbach’s alpha).
- Normality checks and distribution assessments.
- Group comparisons (ANOVA, Kruskal-Wallis, Mann-Whitney U).
- Correlation analysis and regression models.
- Generating descriptive statistics tables and visualizations (e.g., boxplots, heatmaps, regression plots).

🔍 Reproducibility
* Both scripts are designed to be reproducible.
* Input: survey dataset in .csv format.
* Output: cleaned dataset, statistical test results, and figures (saved as .csv or .png depending on the step).

📖 How to Use
1. Place your raw survey data in the project folder (e.g., data/survey_raw.csv).
2. Run preprocess_to_scores.py → generates data/survey_clean.csv.
3. Run Appendix_D_analysis.py → outputs results and figures in results/.

📎 Reference

Full details of the analyses can be found in:

[Factors Influencing Generative AI Adoption in Corporate Tax: A TOE and Trust-Based Perspective], Master’s Thesis, Tilburg University, 2025.
