"""Project KS3 client with per-client private-line proxy; never log credentials."""
import configparser
from pathlib import Path
import boto3
from botocore.config import Config


def client(config_path=Path('outputs/secrets/ks3util-config')):
    parser=configparser.ConfigParser(allow_no_value=True)
    parser.read(config_path)
    config=parser['Credentials']
    return boto3.client('s3',endpoint_url='https://'+config['endpoint'],
        aws_access_key_id=config['accessKeyID'],aws_secret_access_key=config['accessKeySecret'],
        config=Config(signature_version='s3',s3={'addressing_style':'path'},
            proxies={'https':config['proxy'],'http':config['proxy']},
            connect_timeout=10,read_timeout=40,retries={'max_attempts':1}))
