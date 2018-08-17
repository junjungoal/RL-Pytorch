# RL-Pytorch
RL algorithms implemented by Pytorch

- DQN
- Double DQN

## Set up

```
pip install -r requirements.txt
```

## How to Use

```
python main.py -- algorithm_name --env environment_name --learning_rate learning_rate --batch_size batch_size --random_step random_step --log_dir log_dir --weight_dir weight_dir
```

For example....

```
python main.py -- dqn --env Breakout-v4 --learning_rate 0.00025 --batch_size 32 --random_step 50000 --log_dir ./logs/ --weight_dir ./checkpoints/
```

Note that Arguments except for **env** are optional. 

While training or after training, you can check the progress of the training by using TensorBoard. 

```
tensorboard --logdir=./logs/
```

