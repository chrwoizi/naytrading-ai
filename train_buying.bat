python -m pip install numpy
python -m pip install tensorflow
REM python -m pip install matplotlib
python src\train.py --action=buy --model_dir=buying --train_file=buying_train_norm.csv --test_file=buying_test_norm.csv --repeat_train_data=1 --epochs=134
pause