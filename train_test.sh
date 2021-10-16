
# train
python train.py  --config benchmark/mydeeplabv3p.yml \
                 --save_interval 1500 \
                 --save_dir outputs/deeplabv3p_5/ \
                 --do_eval

# test
python predict.py --config benchmark/mydeeplabv3p.yml \
                  --model_path outputs/deeplabv3p_5/best_model/model.pdparams \
                  --image_path ../data/img_testA \
                  --save_dir outputs/deeplabv3p_5/result