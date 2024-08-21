
import os
import pandas as pd 
from pandas import read_csv


file = read_csv("/media/otonom/cold_storage/storage/mide/clinical_followup_357patients.csv")
savepath = ("/media/otonom/cold_storage/storage/mide/TCGA_survival_337patients.csv")

data = pd.DataFrame()
follow = file['case_submitter_id'].tolist()

for i in range (len(follow)):


    x = follow[i]
    clinic = pd.DataFrame(index=[i])
    clinic = clinic.assign(case_id=x)
    clinic = clinic.assign (slide_id=x)

    y = file.loc[file['case_submitter_id']==x,'vital_status'].to_list()

    if y[0] == 'Dead':
        survival_m = file.loc[file['case_submitter_id']==x,'survival_days']
        survival_m = float(survival_m)/30
        clinic=clinic.assign(survival_months=survival_m)
        clinic = clinic.assign(cencorship=1)
        clinic = clinic.assign(label='Dead')
    else:
        survival_m = file.loc[file['case_submitter_id']==x,'last_followup']
        survival_m = float(survival_m)/30
        clinic=clinic.assign(survival_months=survival_m)
        clinic = clinic.assign(cencorship=0)
        clinic = clinic.assign(label = 'Alive')
    
    clinic = clinic.assign(oncotree_code='STAD')
    clinic = clinic.astype(str)
    data = data.append(clinic)
    
print(data.head())
data.to_csv(savepath)
