#!/usr/bin/env python3
"""Stage only the five official TACO archives and two previously licensed MANO files."""
import hashlib
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from ks3_client import client

root=Path.cwd()
assert str(root)=='/share-mixed/projset/eai_pwm/experiments/PWM-benchmark100'
assert shutil.disk_usage(root).free>100*2**30
revision='fa8ed8b4cf5d46949b90ef852128811a2854066a'
files={
 'Egocentric_Camera_Parameters.zip':(18689683,'6e03b9e54d2de0e04bb968a8ec49de65e6737b44bf9b86d6dd3f27a756c01245'),
 'Egocentric_RGB_Videos.zip':(24666017548,'fc2c472091bfb6a170efa681c8eab099fab4f78294ef2dace43ebcb54f618f29'),
 'Hand_Poses.zip':(15231790511,'db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51'),
 'Object_Poses.zip':(34340303,'39c0478395529b550fd218fdbcb9d0c866282c4dfc0d5bac1116461b85e91b64'),
 'Object_Models.zip':(424209827,'7b5209c7e856246cf20b379a599d756d18e8bbbbf7820bf9fe48a334f9a06cfe')}
destination=root/'outputs/data/TACO'
destination.mkdir(parents=True,exist_ok=True)
def download(name):
    size,digest=files[name]
    print(json.dumps({'phase':'download','file':name,'bytes':size}),flush=True)
    path=Path(hf_hub_download(repo_id='leolyliu/TACO',repo_type='dataset',filename=name,
        revision=revision,local_dir=destination,token=False))
    assert path.stat().st_size==size, f'size mismatch: {name}'
    checksum=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''): checksum.update(block)
    assert checksum.hexdigest()==digest, f'hash mismatch: {name}'
    print(json.dumps({'phase':'verified','file':name,'sha256':digest}),flush=True)
    return {'file':name,'bytes':size,'sha256':digest}
api=client()
mano=root/'outputs/data/mano'
mano.mkdir(parents=True,exist_ok=True)
records=[]
for name in ('MANO_LEFT.pkl','MANO_RIGHT.pkl','LICENSE.txt'):
    key='ARCTIC/body_models/mano/'+name
    size=api.head_object(Bucket='baai-eai',Key=key)['ContentLength']
    assert size<5*1024*1024
    path=mano/name
    if not path.exists() or path.stat().st_size!=size:
        temporary=path.with_suffix('.download')
        api.download_file('baai-eai',key,str(temporary))
        assert temporary.stat().st_size==size
        temporary.replace(path)
    records.append({'key':key,'bytes':size,'local_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
with ThreadPoolExecutor(max_workers=3) as pool:
    taco=list(pool.map(download,files))
# The locally inventoried camera archive had an additional _2 suffix. The
# official archive has identical size; preserve its original name and hash.
alias=destination/'Egocentric_Camera_Parameters_2.zip'
if not alias.exists(): alias.symlink_to('Egocentric_Camera_Parameters.zip')
report={'taco_source':'https://huggingface.co/datasets/leolyliu/TACO','revision':revision,
    'taco':taco,'mano':records,'camera_alias':str(alias)}
(root/'outputs/benchmark100_taco_stera_20261029/staging.json').write_text(json.dumps(report,indent=2)+'\n')
print('STAGING_COMPLETE',flush=True)
