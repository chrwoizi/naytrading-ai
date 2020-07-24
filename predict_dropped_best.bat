python -m pip install requests
python -m pip install tensorflow==1.15
cd src
python predict.py --buy_checkpoint_dir=%1\checkpoint\best --sell_checkpoint_dir=%2\checkpoint\best
pause