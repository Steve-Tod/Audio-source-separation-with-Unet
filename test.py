import argparse
from utils.test_utils import source_separation, TestSet

parser = argparse.ArgumentParser(description='Unet source separation test')
parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--file_list', default='./test_info.json', type=str, help='Info json for test')
parser.add_argument('--db', default=True, type=bool, help='Use db as input')
parser.add_argument('--output_audio_dir', default='./result/result_audios/', type=str, help='The dir should exist')
parser.add_argument('--output_json', default='./result/result.json', type=str)
args = parser.parse_args()

ds = TestSet(file_list=args.file_list, db=args.db)
source_separation(args.pretrained_model, ds, args.output_audio_dir, args.output_json)
