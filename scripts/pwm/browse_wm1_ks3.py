#!/usr/bin/env python3
import json
from pathlib import Path
import sys
from ks3_client import client

api=client()
report={}
for prefix in ('','data/','backup_20261023/','ARCTIC/body_models/mano/'):
    response=api.list_objects(Bucket='baai-eai',Prefix=prefix,Delimiter='/',MaxKeys=1000)
    report[prefix]={'prefixes':[x['Prefix'] for x in response.get('CommonPrefixes',[])],
        'objects':[{'key':x['Key'],'bytes':x['Size']} for x in response.get('Contents',[])],
        'truncated':response.get('IsTruncated',False)}
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
