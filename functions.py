import numpy as np
import math
import pandas as pd

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(stock_name1):
	vec = []
	vecvolume = []
	lines = open("data/{}".format(stock_name1) + ".csv", "r").read().splitlines()
	for line in lines[1:]:
		close = line.split(",")
		if close != 'null':
			vec.append(float(line.split(",")[5]))
			vecvolume.append(float(line.split(",")[6]))
	return vec, vecvolume

# returns the sigmoid
def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

# return leaky relu
def leaky_relu(input_value):
	if input_value > 0:
		return input_value
	else:
		return 0.05 * input_value

# returns an an n-day state representation ending at time t
def getState(data,datavolume, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []

	# 현재-전날 가격
	for i in range(n - 5):
		#res.append(leaky_relu(block[i + 1] - block[i]))
		res.append((block[i]))

	# 상태 추가(MA)
	#res.append(leaky_relu(datavolume[t]))
	# MA
	res.append(getMA(data, t, 5))
	res.append(getMA(data, t, 20))
	res.append(getMA(data, t, 60))
	res.append(getMA(data, t, 120))
	return np.array([res])

def getMA(data, t, n):
	d = t - n + 1
	# 데이터범위 넘어가면
	while d + n > len(data):
		n -= 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	MA = 0
	for i in range(n):
		MA += block[i]
	return MA/n
