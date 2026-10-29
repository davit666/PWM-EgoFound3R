#!/usr/bin/env python3
import json
from pathlib import Path
import sys
from ks3_client import client

api=client()
report={}
for prefix in (sys.argv[2:] or ('','data/','backup_20261023/','ARCTIC/body_models/mano/')):
    item={'prefixes':[], 'objects':[], 'truncated':True}
    report[prefix]=item
    marker=''
    for page in range(100):
        response=api.list_objects(Bucket='baai-eai',Prefix=prefix,Delimiter='/',MaxKeys=1000,Marker=marker)
        item['prefixes'].extend(x['Prefix'] for x in response.get('CommonPrefixes',[]))
        item['objects'].extend({'key':x['Key'],'bytes':x['Size']} for x in response.get('Contents',[]))
        item['truncated']=response.get('IsTruncated',False)
        Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
        if not item['truncated']: break
        next_marker=response.get('NextMarker')
        if not next_marker or next_marker==marker: break
        marker=next_marker
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
