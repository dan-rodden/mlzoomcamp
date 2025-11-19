# MLZoomcamp Midterm Project: Healthy Control Platlet Binary Classification


This is the dataset I am using. Finally!: <link src="https://www.kaggle.com/datasets/samiraalipour/gene-expression-omnibus-geo-dataset-gse68086?resource=download">

Includes Tumor-Educated Platlets (TEPs) with healthy (wt) blood platelet cells. For binary classification, goal is to distinguish the difference between TEPS and healthy blood cells with the hope that that features can be found in blood providing some biomarkers.

The target variable used is `!Sample_characteristics_ch1.3`. With this encoding. `"cancer type: HC"` stands for healthy controls used in the dataset. Other samples are classified as cancer in the `target` columns of the dataframe.
- "cancer type: HC"                55
- "cancer type: GBM"               40
- "cancer type: Lung"              40
- "cancer type: CRC"               38
- "cancer type: Pancreas"          35
- "cancer type: Breast"            24
- "cancer type: Hepatobiliary"     13
- "mutational subclass: wt"        12
- "mutational subclass: KRAS"      11
- "mutational subclass: EGFR"       9
- "mutational subclass: HER2+"      6
- "mutational subclass: PIK3CA"     2



For binary classification the target column is encoded via:
- 0: Cancerous
- 1: Healthy Control Blood Platlet










