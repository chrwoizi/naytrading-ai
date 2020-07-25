python -m pip install numpy
python -m pip install tensorflow==1.15
python -m pip install tensorflow-gpu==1.15
REM python -m pip install matplotlib
python src\train.py --action=sell --model_dir=selling --train_file=selling_train_norm.csv --test_file=selling_test_norm.csv --additional_columns=1 --epochs=183
pause