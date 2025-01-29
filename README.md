# hyperspectral-selection
Hyperspectral bands selection and experiments framework

To run the code:
1.python -m visdom.server
2.python main_cross_val.py --model hamida_fs --dataset Salinas --training_sample 1 --patch_size 5 --epoch 200 --epoch_second 0 --cuda 0 --lr 0.002 --bands_amount 9 --batch_size 256  --reset_gates -1 --lam 1.1

