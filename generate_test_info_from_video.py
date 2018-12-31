import argparse
from utils.test_utils import generate_info_for_audio
from utils.Judge import getInstrumentPos

parser = argparse.ArgumentParser(description='Find instrument and position')
parser.add_argument(
    '--video_dir',
    default='/data/data_61WwmOjr/std/testset25/testvideo/',
    type=str)
parser.add_argument(
    '--audio_dir',
    default='/data/data_61WwmOjr/std/testset25/gt_audio/',
    type=str)
parser.add_argument('--output', default='./info/test_info.json', type=str)
args = parser.parse_args()

generate_info_for_audio(args.video_dir, args.audio_dir, getInstrumentPos,
                        args.output)
