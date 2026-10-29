#!/usr/bin/env python3
"""Read small annotation/schema samples; no bulk extraction or array loading."""
import json
from pathlib import Path
import sys
import tarfile
import zipfile
import h5py

base = Path('/share-mixed/project/eai_pwm/datasets')
report = {}
for name, relative in {'TACO_hands':'TACO/Hand_Poses.zip', 'TACO_cameras':'TACO/Egocentric_Camera_Parameters_2.zip',
                       'TACO_rgb':'TACO/Egocentric_RGB_Videos.zip', 'TACO_objects':'TACO/Object_Poses.zip'}.items():
    path = base / relative
    try:
        with zipfile.ZipFile(path) as file:
            entries = [x for x in file.infolist() if not x.is_dir()]
            result = {'count':len(entries),'members':[{'name':x.filename,'bytes':x.file_size} for x in entries[:12]]}
            for item in entries[:3]:
                if item.file_size < 500000 and Path(item.filename).suffix in ('.json','.txt','.csv'):
                    result.setdefault('text_samples',{})[item.filename] = file.read(item).decode(errors='replace')[:10000]
            report[name] = result
    except Exception as e:
        report[name] = {'error':str(e)}
path = base / 'H2O/manolabel_v1.1.tar.gz'
try:
    with tarfile.open(path,'r|gz') as file:
        report['H2O_mano'] = []
        for item in file:
            report['H2O_mano'].append({'name':item.name,'bytes':item.size})
            if item.isfile() and item.size < 20000 and item.name.endswith('.txt'):
                report['H2O_mano'][-1]['text'] = file.extractfile(item).read().decode()[:4000]
            if len(report['H2O_mano']) == 15:
                break
except Exception as e:
    report['H2O_error'] = str(e)
session = Path('/share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905')
with h5py.File(session/'annotation.hdf5','r') as file:
    report['STERA_schema'] = {}
    def visit(name,obj):
        if isinstance(obj,h5py.Dataset):
            item = {'shape':list(obj.shape),'dtype':str(obj.dtype)}
            if obj.dtype.kind in 'fiu' and obj.size and obj.ndim and obj.size/obj.shape[0] <= 256:
                item['first'] = obj[:1].tolist()
            report['STERA_schema'][name] = item
    file.visititems(visit)
report['calibrations'] = {}
for file in (session/'calibrations').glob('*'):
    report['calibrations'][file.name] = file.read_text()[:12000] if file.suffix in ('.json','.txt') and file.stat().st_size < 100000 else file.stat().st_size
report['HOI4D_candidates'] = {str(p):p.is_dir() for p in [base/'HOI4D',base/'raw/HOI4D',base/'raw/data/HOI4D',Path('/share-mixed/project/eai_pwm/raw/data/HOI4D')]}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
