python train.py --do-eval --eval-datasets race --sub-file val_submission.csv --eval-dir $1 --save-dir $2 >> log_$3.txt
python train.py --do-eval --eval-datasets relation_extraction --sub-file val_submission.csv --eval-dir $1 --save-dir $2 >> log_$3.txt
python train.py --do-eval --eval-datasets duorc --sub-file val_submission.csv --eval-dir $1 --save-dir $2 >> log_$3.txt
