import json

CONFIG_FILE = 'config.json'
print('Loading {}'.format(CONFIG_FILE))
config = json.load(open(CONFIG_FILE))
