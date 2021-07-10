# transfer-learning


# train: 
python train_mobilenet.py --config config/config_mobilenet.yaml

# val:
python test.py --weight path/to/weight --config config/config_test.yaml
python test_simple.py --weight path/to/weight --config config/config_test.yaml

# test object class
python test_mask_detection.py --weight path/to/weight --csv_file path/to/file/test

