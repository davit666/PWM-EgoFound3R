#!/usr/bin/env python3
"""Import existing authorized credentials through hidden stdin, never argv/logs."""
import json
from pathlib import Path
import sys
print('WAITING_FOR_KS3_CREDENTIALS',flush=True)
data=json.loads(sys.stdin.readline())
for key in ('access_key_id','secret_access_key'):
    if not data.get(key) or '\n' in data[key] or '\r' in data[key]:
        raise ValueError('invalid credential input')
path=Path('outputs/secrets/ks3util-config')
path.write_text('[Credentials]\nlanguage\naccessKeyID='+data['access_key_id']+'\naccessKeySecret='+data['secret_access_key']+
    '\nendpoint=ks3-cn-beijing-internal.ksyuncs.com\nloglevel\nproxy=http://10.8.36.21:2080\n')
path.chmod(0o600)
print('KS3_CREDENTIALS_IMPORTED',flush=True)
