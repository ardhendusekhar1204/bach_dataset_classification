import os
import pandas as pd
top='/home/Drive4/drive2/ardhendubapu/Bach/'
data_record={'path':[],'slide_name':[],'label':[]}
for root, directories, files in os.walk(top, topdown=False):
    for name in files:
        data_record['slide_name'].append(name)
        data_record['path'].append(os.path.join(root,name))
        data_record['label'].append(root.split('/')[-1])
df=pd.DataFrame(data_record)
df.label[df.label=='Normal']=0
df.label[df.label=='InSitu']=1
df.label[df.label=='Benign']=2
df.label[df.label=='Invasive']=3
df.to_csv('bach_data.csv',index=False)   
