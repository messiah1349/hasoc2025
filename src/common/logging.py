import logging
import logging.config
import json
import os

def setup_logging(
    default_path='logging.json',
    default_level='WARNING',
    env_key='LOG_CFG'
):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    # logging.getLogger("httcore.connection").setLevel(logging.INFO)
    # logging.getLogger("telegram.ext.ExtBot").setLevel(logging.INFO)
