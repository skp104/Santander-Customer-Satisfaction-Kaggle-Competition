import pandas as pd
from XG_Boost import XGB
from Random_Forest import RF

XGB_Dataframe = XGB()
RF_Dataframe = RF()

IDs = XGB_Dataframe["ID"]
XGB = XGB_Dataframe["TARGET"]
RF = RF_Dataframe["TARGET"]

data_pts = []
final = []
for each in IDs:
    data_pts.append(each)
    
for i in range(0,len(IDs)):
    xg_boost = 0.9*XGB[i]
    random_forest = 0.1*RF[i]
    final.append(xg_boost+random_forest)
     
print "making submissions"    
submission = pd.DataFrame({"ID":data_pts, "TARGET":final})
submission.to_csv("submission_XGB_Ensemble.csv", index=False)