rm ./result/result_audios/*.wav 
python test.py --pretrained_model $1 --db $2
python Evaluate.py $3
