from argparse import ArgumentParser
from modeler.main import modeler
import json


def parse_args():
    parser = ArgumentParser(description="Description of your program")
    parser.add_argument("-m", "--model-id", help="Hugging Face Model ID", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    r = modeler(args.model_id)
    return json.dumps(r, default=lambda o: o.__dict__, indent=2)
