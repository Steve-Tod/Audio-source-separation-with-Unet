rm ./result/result_audios/*.wav 
python test.py --pretrained_model $1
python Evaluate.py
