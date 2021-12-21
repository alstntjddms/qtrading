import sys
import torch
from agent.agent import Agent
from functions import *
from tqdm import tqdm
import yfinance as yf
import time

start = time.time()

def average_buy_price(inventory1):
	average = 0
	for i in inventory1:
		average += i

	if len(inventory1) > 0:
		return average / len(inventory1)
	else:
		return 0

if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# 데이터로드
stock_name1 = stock_name
stock_name = yf.download(stock_name,
                      start='2021-1-15',
                      end='2021-12-15')
stock_name = stock_name.reset_index()
stock_name['Date'] = pd.to_datetime(stock_name['Date'])
stock_name.set_index('Date', inplace=True)
stock_name.to_csv('data/{}.csv'.format(stock_name1), mode='w', header=True)

agent = Agent(window_size)
data, datavolume = getStockDataVec(stock_name1)
#volume = volume(stock_name)
l = len(data) - 1
x_data = range(l)
o = 0

print("시작")
model_count = 0
for e in tqdm(range(episode_count + 1)):
	print("Episode " + str(e) + "/" + str(episode_count))

	state = getState(data, datavolume, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []
	p_data = 0
	buyact = 0
	sellact = 0
	sitact = 0
	sitsell =0
	closes = []
	buys = []
	sells = []
	sit = []

	for t in range(l):
		closes.append(data[t])
		action = agent.act(state)
		next_state = getState(data,datavolume, t + 1, window_size + 1)
		reward = 0

		#print(action)
		if action == 1: # buy
			agent.inventory.append(data[t])
			#print("Buy: " + formatPrice(data[t]))
			reward = data[t+1] - data[t]
			buyact +=1

			# 시각화 리스트 추가
			buys.append(data[t])
			sells.append(None)
			sit.append(None)

			#Buy에 리워드를 어떻게 줄것인가????
			#지금코드는 사고고 안팔려고하거나 sit만함(수정해야함)

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward =data[t] - average_buy_price(agent.inventory)
			total_profit += (data[t] - bought_price)
			#print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			#print("Buy: " + formatPrice(data[t]) + "Profit: " + format(data[t] - bought_price))
			sellact += 1

			# 시각화 리스트 추가
			buys.append(None)
			sit.append(None)
			sells.append(data[t])

		elif action == 0 : # sit
			reward = data[t+1] - data[t]
			#reward = np.tanh((data[t] - average_buy_price(agent.inventory)) * len(agent.inventory))
			sitact += 1

			# 시각화 리스트 추가
			buys.append(None)
			sells.append(None)
			sit.append(data[t])

		elif len(agent.inventory) == 0 and action == 2:
			#reward = -55555
			sitsell +=1

			# 시각화 리스트 추가
			buys.append(None)
			sells.append(None)
			sit.append(data[t])

		if p_data % 10 == 0 and p_data > 0:
			#print('1년 매매일지', 'Buy횟수' ,buyact, "Sell횟수", sellact, "Sit횟수", sitact, "주식수", buyact-sellact, "총이익", total_profit, "매수평균", average_buy_price(agent.inventory))
			#print('Buy횟수{}, "Sell횟수{}, Sit횟수{], 주식수{}, 이익{}, 매수평균{}'
			#	  .format(buyact, sellact, sitact, buyact - sellact, total_profit, average_buy_price(agent.inventory)))
			p_data = 0
		p_data += 1
		#print("인벤토리 평균값" + format(average_buy_price(agent.inventory)))
		done = True if t == l - 1 else False
		agent.memory.push(state, action, next_state, reward)
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + format(total_profit))
			print('1년 매매일지', 'Buy횟수' ,buyact, "Sell횟수", sellact, "Sit횟수", sitact, "sitsell", sitsell, "주식수", buyact-sellact, "총이익", (data[t] - average_buy_price(agent.inventory))*len(agent.inventory)+total_profit, "매수평균", average_buy_price(agent.inventory), "종가", data[t] )
			print("--------------------------------")

		agent.optimize()

	# 10epoch마다 model_state저장
	if e % 10 == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())
		torch.save(agent.policy_net, 'models/policy_model1')
		torch.save(agent.target_net, 'models/target_model1')
		#torch.save(agent.policy_net, 'models/policy_model_%d' %(model_count))
		#torch.save(agent.target_net, 'models/target_model_%d' %(model_count))
		model_count += 1

	import matplotlib.pyplot as plt

	if e % 200 == 0:
		plt.figure(figsize=(12, 12))
		plt.plot(x_data, closes)
		plt.plot(x_data, buys, marker='*', markersize=7, markerfacecolor='r')
		plt.plot(x_data, sells, marker='*', markersize=7, )
		plt.plot(x_data, sit, marker='*', markersize=7, )

		plt.title(stock_name1)
		plt.legend(['close', 'buy_num%d'%(buyact), 'sell_num%d'%(sellact+sitsell), 'sit_num%d'%(sitact)], loc='upper left')
		plt.savefig('images/Result%d.png' %(o), dpi=300)
		o += 1
print("%d학습하는데 총 소요시간 :"%(episode_count), time.time() - start, '초')
#plt.show()