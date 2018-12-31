import os, json, argparse, random, itertools

parser = argparse.ArgumentParser(description='Generate data list json file')
parser.add_argument(
    '--audio_root',
    type=str,
    help='Root dir of audio files, like /data/trainset/audios/solo/')
parser.add_argument(
    '--train_list',
    default='./info/train_list.json',
    type=str,
    help='output train list file')
parser.add_argument(
    '--test_list',
    default='./info/test_list.json',
    type=str,
    help='output test list file')

args = parser.parse_args()

if __name__ == '__main__':
    train_list = {}
    test_list = {}

    instruments = tuple(os.listdir(args.audio_root))
    combinations = list(itertools.combinations(instruments, 2))
    for comb in combinations:
        instrument1, instrument2 = comb
        comb_name = instrument1 + "-" + instrument2
        train_list[comb_name] = []
        test_list[comb_name] = []

        instrument1_seg = list(
            os.listdir(os.path.join(args.audio_root, instrument1)))
        instrument2_seg = list(
            os.listdir(os.path.join(args.audio_root, instrument2)))
        seg_product = list(itertools.product(instrument1_seg, instrument2_seg))
        selected_pairs = random.sample(seg_product, 790)
        random.shuffle(selected_pairs)
        for i, pair in enumerate(selected_pairs):
            wav1_path = os.path.join(instrument1, pair[0])
            wav2_path = os.path.join(instrument2, pair[1])
            if i < 750:
                train_list[comb_name].append({
                    "wav1_path": wav1_path,
                    "wav2_path": wav2_path
                })
            else:
                test_list[comb_name].append({
                    "wav1_path": wav1_path,
                    "wav2_path": wav2_path
                })

    with open(args.train_list, 'w') as f:
        json.dump(train_list, f)
    with open(args.test_list, 'w') as f:
        json.dump(test_list, f)
