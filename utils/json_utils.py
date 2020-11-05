import json


def write_json(json_dict, json_file):
    with open(json_file, 'w') as f:
        json.dump(json_dict, f, indent=2)
        f.close()


def read_json(file_path):
    with open(file_path, 'r') as f:
        val_dict = json.load(f)
        f.close()
        return val_dict
