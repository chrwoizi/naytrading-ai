pip install requests
pip install tensorflow
cd src
python predict.py --buy_checkpoint_dir=%1\checkpoint --sell_checkpoint_dir=%2\checkpoint
pause