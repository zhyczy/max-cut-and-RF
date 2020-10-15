import sys
import math
import numpy as np
import random

state_space = {}
state_cut_recor = {}
ROUND_NUM = 2000
DEFAULT_WEIGHT = 2
RANK = 1
K1 = 1
K2 = 2

class State(object):

	def __init__(self):
		self.setA = []
		self.setB = []
		self.dic = {}
		self.transistion_probability = []
		self.value = 0
		self.as_child_value = 0

	def set_end(self,end):
		self.cut_end = end

	def get_setA(self):
		return self.setA

	def set_setA(self, setA):
		self.setA = setA.copy()

	def get_setB(self):
		return self.setB

	def set_setB(self, setB):
		self.setB = setB.copy()

	def get_dic(self):
		return self.dic

	def set_dic(self, dic):
		self.dic = dic.copy()

	def get_value(self):
		return self.value

	def set_value(self,rewards):
		self.value = rewards

	def get_as_child_value(self):
		return self.as_child_value

	def set_as_child_value(self,rewards):
		self.as_child_value = rewards

	def get_transistion(self):
		return self.transistion_probability

	def set_transistion(self,action_and_prob):
		self.transistion_probability=action_and_prob.copy()

	def __repr__(self):
		return "State: {}".format(
		hash(self))

class Node(object):

	def __init__(self):
 		self.state = None

	def set_state(self, state):
		self.state = state

	def get_state(self):
		return self.state

def evaluation_function(setA,setB,dic):
	possible_Cut = 0
	current_Cut = 0  
	for k in dic.keys():
		if dic[k]==0:
			if k[0] in setA and k[1] in setA:
				continue
			elif k[0] in setB and k[1] in setB:
				continue
			else:
				possible_Cut +=1
		else:
			current_Cut +=1
	rewards = K1*current_Cut + K2*possible_Cut 
	return [rewards,current_Cut]

def fill_distribution(Node):
	its_State=Node.get_state()
	dic = its_State.get_dic()
	setA = its_State.get_setA()
	setB = its_State.get_setB()

	sep_sum = 0
	state_key = dic_encoding(dic)

	act_prob = []
	for k in dic.keys():
		if dic[k]==0:
			#rule out exist simutaneously 7
			if k[0] in setA and k[1] in setA:
				continue
			elif k[0] in setB and k[1] in setB:
				continue
			else:
				temp_dic = dic.copy() 
				temp_A = setA.copy()
				temp_B = setB.copy()
				if k[0] in setA and k[1] in setB:
					temp_dic[k] = 1
				elif k[0] in setB and k[1] in setA:
					temp_dic[k] = 1
				elif k[0] in setA and k[1] not in setB:
					temp_dic[k] = 1
					temp_B.append(k[1])
				elif k[0] in setB and k[1] not in setA:
					temp_dic[k] = 1
					temp_A.append(k[1])
				elif k[1] in setA and k[0] not in setB:
					temp_dic[k] = 1
					temp_B.append(k[0])
				elif k[1] in setB and k[0] not in setA:
					temp_dic[k] = 1
					temp_A.append(k[0])
				elif k[0] not in setA and k[0] not in setB and k[1] not in setA and k[1] not in setB:
					temp_dic[k] = 1
					ut = random.uniform(0,1)
					if ut >= 0.5:
						temp_A.append(k[0])
						temp_B.append(k[1])
					else:
						temp_A.append(k[1])
						temp_B.append(k[0])

				state_key = dic_encoding(temp_dic)
				if state_key in state_space.keys():
					temp_state = state_space[state_key]
					as_child_reward = temp_state.get_as_child_value()
				else:
					reward=evaluation_function(temp_A,temp_B,temp_dic)
					self_reward = reward[1]
					as_child_reward = reward[0]

					temp_state = State()
					temp_state.set_value(self_reward)
					temp_state.set_as_child_value(as_child_reward)
					temp_state.set_setA(temp_A)
					temp_state.set_setB(temp_B)
					temp_state.set_dic(temp_dic)
					state_space[state_key] = temp_state

				sep_sum += as_child_reward
				act_prob.append([temp_state,as_child_reward])
		else:
			temp_dic = dic.copy() 
			temp_A = setA.copy()
			temp_B = setB.copy()
			if k[0] in temp_A:
				temp_A.remove(k[0])
			if k[0] in temp_B:
				temp_B.remove(k[0])
			if k[1] in temp_A:
				temp_A.remove(k[1])
			if k[1] in temp_B:
				temp_B.remove(k[1])
			temp_dic[k]=0
			state_key = dic_encoding(temp_dic)

			if state_key in state_space.keys():
				temp_state = state_space[state_key]
				as_child_reward = temp_state.get_as_child_value()
			else:
				reward=evaluation_function(temp_A,temp_B,temp_dic)
				self_reward = reward[1]
				as_child_reward = reward[0]

				temp_state = State()
				temp_state.set_value(self_reward)
				temp_state.set_as_child_value(as_child_reward)
				temp_state.set_setA(temp_A)
				temp_state.set_setB(temp_B)
				temp_state.set_dic(temp_dic)
				state_space[state_key] = temp_state

			sep_sum += as_child_reward
			act_prob.append([temp_state,as_child_reward])
	for entry in act_prob:
		entry[1] = entry[1]/sep_sum
	Node.get_state().set_transistion(act_prob)
	return act_prob

def dic_encoding(dic):
	dic_to_list = [(k,v) for k,v in dic.items()]
	state_key = tuple(dic_to_list)
	return state_key

def find_counterpart(Node,rival):
	rival_key = dic_encoding(rival)
	cur_state = Node.get_state()
	if len(cur_state.get_transistion())==0:
		prob_distribution = fill_distribution(Node)
	else:
		prob_distribution = cur_state.get_transistion()
	
	tr = 1
	for tran in prob_distribution:
		tran_dic = tran[0].get_dic()
		tran_key = dic_encoding(tran_dic)
		if tran_key == rival_key:
			tr = tran[1]
			break
	di = reward_to_prob(cur_state.get_value())
	return (tr,di)

def reward_to_prob(reward):
	return (1 - math.exp(-(reward+1)))

def roulette(prob_distribution):
	roll = random.uniform(0.0,1.0)
	prob_sum = 0
	for v in prob_distribution:
		prob_sum += v[1]
		if roll<=prob_sum:
			return v

def throw_coin(d1,p1_2,d2,p2_1):
	mh = (d2*p2_1)/(d1*p1_2)
	accept = min(mh,1)
	coin = random.uniform(0.0,1.0)
	if coin<=accept:
		return True
	else:
		return False

def calculate_cut(dic):
	cut_number = 0
	for kk in dic.keys():
		if dic[kk]==1:
			cut_number +=1
	return cut_number

def main():
#	total_point=[x for x in range(1,21)]
	point_num = 20
	edge_num = 30
	total_edge=[(1,2),(1,5),(2,3),(3,4),(4,5),
				(1,16),(2,15),(3,13),(4,8),(5,6),
(15,17),(14,15),(13,14),(11,13),(8,11),(7,8),(6,7),(6,18),(16,18),(16,17),
				(17,20),(12,14),(10,11),(7,9),(18,19),
				(19,20),(9,19),(9,10),(10,12),(12,20)]
	dic = {}
	for i in total_edge:
		dic[i]=0
	state_key = dic_encoding(dic)

	prev_node = Node()
	init_state = State()
	init_state.set_setA([])
	init_state.set_setB([])
	init_state.set_dic(dic)

	prev_node.set_state(init_state)
	prev_reward = evaluation_function([],[],dic)
	prev_self = prev_reward[1]
	prev_chil = prev_reward[0]
	prev_dic = dic.copy()

	init_state.set_value(prev_self)
	init_state.set_as_child_value(prev_chil)
	state_space[state_key] = init_state

	for Round in range(ROUND_NUM):
		cut_number = calculate_cut(prev_dic)
		prev_state_key = dic_encoding(prev_dic)
		if prev_state_key not in state_cut_recor.keys():
			state_cut_recor[prev_state_key] = cut_number
		else:
			state_cut_recor[prev_state_key] = max(state_cut_recor[prev_state_key],cut_number)

		prob_distribution = fill_distribution(prev_node)
		rollout = roulette(prob_distribution)
		state_proposal = rollout[0]
		p1_2 = rollout[1]
		d1 = reward_to_prob(prev_self)

		new_node = Node()
		new_node.set_state(state_proposal)
		counterpart = find_counterpart(new_node,prev_dic)
		p2_1 = counterpart[1]
		d2 = counterpart[0]
		accept_proposal = throw_coin(d1,p1_2,d2,p2_1)

		if accept_proposal:
			prev_self = state_proposal.get_value()
			prev_node = new_node
			prev_dic = state_proposal.get_dic()

	ran = []
	for it in range(RANK):
		max_cut = 0
		max_dict = {}
		max_key = []
		setA = []
		setB = []
		for k,v in state_cut_recor.items():
			cur_dic = state_space[k].get_dic()
			cur_key = dic_encoding(cur_dic)
			if int(v) > max_cut and cur_key not in ran:
				max_cut = v
				max_dict = cur_dic
				max_key = cur_key
				setA = state_space[k].get_setA()
				setB = state_space[k].get_setB()
		ran.append(cur_key)
		print("Rank: {}, Cut_number:{}, Dic:{}, Set A:{}, Set B:{}".format(
		it, max_cut , max_dict, setA, setB)
		)
main()

# Rank: 0, Cut_number:24, 
# Dic:{(1, 2): 1, (1, 5): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (1, 16): 1, (2, 15): 1, (3, 13): 1, 
# (4, 8): 0, (5, 6): 0, (15, 17): 1, (14, 15): 1, (13, 14): 1, (11, 13): 1, (8, 11): 1, (7, 8): 1,
#  (6, 7): 1, (6, 18): 0, (16, 18): 0, (16, 17): 1, (17, 20): 1, (12, 14): 1, (10, 11): 1, (7, 9): 0,
#   (18, 19): 1, (19, 20): 1, (9, 19): 0, (9, 10): 1, (10, 12): 1, (12, 20): 1}, 
# Set A:[2, 3, 5, 11, 16, 20, 9, 14, 7, 17], Set B:[1, 4, 8, 10, 13, 12, 6, 15]