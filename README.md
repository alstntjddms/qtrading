# Q-Trader

** Use in your own risk **

Pytorch implmentation from q-trader(https://github.com/edwardhdlu/q-trader)


## Running the Code

To train the model, download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`
```
mkdir models
python train BTC-USD 10 1000
or
python train XRP-USD 10 1000
```


## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code
