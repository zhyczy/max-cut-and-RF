import sys
import math
import numpy as np
import random

AVAILABLE_CHOICES = []
AVAILABLE_CHOICE_NUMBER = 0
POINT_NUMBER = 20
EDGE_NUMBER = 30

setA_copy=[]
setB_copy=[]
setA=[]
setB=[]

dic={}
dic_copy={}

class State(object):

	def __init__(self):
		self.current_value = 0.0
		self.cumulative_choices = []
		self.cut_end = 0

	def check_end(self):
		return self.cut_end

	def set_end(self,end):
		self.cut_end = end

	def get_current_value(self):
		return self.current_value

	def set_current_value(self, value):
		self.current_value = value

	def get_cumulative_choices(self):
		return self.cumulative_choices

	def set_cumulative_choices(self, choices):
		self.cumulative_choices = choices

	def is_terminal(self):
		if self.cut_end == 1:
			return True
		else:
			return False 

	def get_next_state_with_random_choice(self):
		random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])
		next_state = State()
		next_state.set_current_value(self.current_value)
		next_state.set_cumulative_choices(random_choice)
		return next_state

	def __repr__(self):
		return "State: {}, value: {}, round: {}, choices: {}".format(
		hash(self), self.current_value, self.current_round_index,
		self.cumulative_choices)

class Node(object):

	def __init__(self):
 		self.parent = None
 		self.children = []
 		self.visit_times = 0
 		self.quality_value = 0.0
 		self.state = None

	def set_state(self, state):
		self.state = state

	def get_state(self):
		return self.state

	def get_parent(self):
		return self.parent

	def set_parent(self, parent):
		self.parent = parent

	def get_children(self):
		return self.children

	def get_visit_times(self):
		return self.visit_times

	def set_visit_times(self, times):
		self.visit_times = times

	def visit_times_add_one(self):
		self.visit_times += 1

	def get_quality_value(self):
		return self.quality_value

	def set_quality_value(self, value):
		self.quality_value = value

	def quality_value_add_n(self, n):
		self.quality_value += n
	
	def is_all_expand(self):
		return len(self.children) == AVAILABLE_CHOICE_NUMBER

	def add_child(self, sub_node):
		sub_node.set_parent(self)
		self.children.append(sub_node)

	def __repr__(self):
		return "Node: {}, Q/N: {}/{}, state: {}".format(
		hash(self), self.quality_value, self.visit_times, self.state)

def MCTS():
	# Create the initialized state and initialized node
	init_state = State()
	init_node = Node()
	init_node.set_state(init_state)
	current_node = init_node
	current_node = monte_carlo_tree_search(current_node)
	return current_node.get_state()

def monte_carlo_tree_search(node):
	computation_budget = 800
	for i in range(computation_budget):   
		# 1. Find the best node to expand
		expand_node = tree_policy(node)
		# 2. Random run to add node and get reward
		reward = rollout_policy(expand_node)
		# 3. Update all passing nodes with reward
		backup(expand_node, reward)
	# N. Get the best next node
	best_next_node = best_child(node, False)
	return best_next_node

def tree_policy(node):
	global setA_copy
	global setB_copy
	global cut_copy
	global dic_copy
	global AVAILABLE_CHOICES
	global AVAILABLE_CHOICE_NUMBER
	setA_copy = setA.copy()
	setB_copy = setB.copy()
	# cut_copy = cut_edge.copy()
	dic_copy = dic.copy()
	choices = []

	for k in dic_copy.keys():
		if dic_copy[k]==0:
			#rule out exist simutaneously 7
			if k[0] in setA_copy and k[1] in setA_copy:
				continue
			elif k[0] in setB_copy and k[1] in setB_copy:
				continue
			else:
				temp_dic = dic.copy() 
				temp_A = setA.copy()
				temp_B = setB.copy()
				if k[0] in setA_copy and k[1] in setB_copy:
					temp_dic[k] = 1
				elif k[0] in setB_copy and k[1] in setA_copy:
					temp_dic[k] = 1
				elif k[0] in setA_copy and k[1] not in setB_copy:
					temp_dic[k] = 1
					temp_B.append(k[1])
				elif k[0] in setB_copy and k[1] not in setA_copy:
					temp_dic[k] = 1
					temp_A.append(k[1])
				elif k[1] in setA_copy and k[0] not in setB_copy:
					temp_dic[k] = 1
					temp_B.append(k[0])
				elif k[1] in setB_copy and k[0] not in setA_copy:
					temp_dic[k] = 1
					temp_A.append(k[0])
				elif k[0] not in setA_copy and k[0] not in setB_copy and k[1] not in setA_copy and k[1] not in setB_copy:
					temp_dic[k] = 1
					ut = random.uniform(0,1)
					if ut >= 0.5:
						temp_A.append(k[0])
						temp_B.append(k[1])
					else:
						temp_A.append(k[1])
						temp_B.append(k[0]) 
				choices.append([temp_A,temp_B,temp_dic])

	AVAILABLE_CHOICES = choices
	AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
# Check if the current node is the leaf node
	while node.get_state().is_terminal() == False:
		if node.is_all_expand():
			node = best_child(node, True)
			choices = []
			next_state = node.get_state()
			policy_choice = next_state.get_cumulative_choices()
			setA_copy = policy_choice[0].copy()
			setB_copy = policy_choice[1].copy()
			dic_copy = policy_choice[2].copy()

			for k in dic_copy.keys():
				if dic_copy[k]==0:
					#rule out exist simutaneously 7
					if k[0] in setA_copy and k[1] in setA_copy:
						continue
					elif k[0] in setB_copy and k[1] in setB_copy:
						continue
					else:
						temp_dic = dic.copy() 
						temp_A = setA.copy()
						temp_B = setB.copy()
						if k[0] in setA_copy and k[1] in setB_copy:
							temp_dic[k] = 1
						elif k[0] in setB_copy and k[1] in setA_copy:
							temp_dic[k] = 1
						elif k[0] in setA_copy and k[1] not in setB_copy:
							temp_dic[k] = 1
							temp_B.append(k[1])
						elif k[0] in setB_copy and k[1] not in setA_copy:
							temp_dic[k] = 1
							temp_A.append(k[1])
						elif k[1] in setA_copy and k[0] not in setB_copy:
							temp_dic[k] = 1
							temp_B.append(k[0])
						elif k[1] in setB_copy and k[0] not in setA_copy:
							temp_dic[k] = 1
							temp_A.append(k[0])
						elif k[0] not in setA_copy and k[0] not in setB_copy and k[1] not in setA_copy and k[1] not in setB_copy:
							temp_dic[k] = 1
							ut = random.uniform(0,1)
							if ut >= 0.5:
								temp_A.append(k[0])
								temp_B.append(k[1])
							else:
								temp_A.append(k[1])
								temp_B.append(k[0]) 
						choices.append([temp_A,temp_B,temp_dic])

			AVAILABLE_CHOICES = choices
			AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
			if AVAILABLE_CHOICE_NUMBER == 0:
				node.get_state().set_end(1)
		else:
			sub_node = expand(node)
			return sub_node
	setA_copy = setA.copy()
	setB_copy = setB.copy()
	dic_copy = dic.copy()
	# Return the leaf node
	return node

def expand(node):
	tried_sub_node_states = [sub_node.get_state() for sub_node in node.get_children()]
	new_state = node.get_state().get_next_state_with_random_choice()
	# Check until get the new state which has the different action from others
	while new_state in tried_sub_node_states:
		new_state = node.get_state().get_next_state_with_random_choice()
	sub_node = Node()
	sub_node.set_state(new_state)
	node.add_child(sub_node)
	return sub_node

#evaluatation function
def rollout_policy(node):
	current_state = node.get_state()
	policy_choice = current_state.get_cumulative_choices()
	final_state_reward = evaluation_function(policy_choice[0],policy_choice[1],policy_choice[2])
	return final_state_reward

def rollout_policy_tail(node):
	current_state = node.get_state()
	while current_state.is_terminal()==False:
		current_state=current_state.get_next_state_with_random_choice()
	policy_choice = current_state.get_cumulative_choices()
	final_state_reward = evaluatation_function(policy_choice[0],policy_choice[1],policy_choice[2])
	return final_state_reward

def backup(node, reward):
	while node != None:
		node.visit_times_add_one()
		node.quality_value_add_n(reward)
		node = node.parent

def graph_cut_finish(setA,setB,dic):
	split_num = len(setA) + len(setB)
	if split_num == POINT_NUMBER:
		return True
	else:
		return False

def evaluation_function(setA,setB,dic):
	possible_Cut = 0
	current_Cut = 0  
	for k in dic.keys():
		if dic[k]==0:
			#rule out exist simutaneously 7
			if k[0] in setA and k[1] in setA:
				continue
			elif k[0] in setB and k[1] in setB:
				continue
			else:
				possible_Cut +=1
		else:
			current_Cut +=1
	rewards = 0.5*current_Cut + 2*possible_Cut 
	# 2:1 10 times average 18.5
	# 1:2 10 times average 19.1
	return rewards

def best_child(node, is_exploration):
	# TODO: Use the min float value
	best_score = -sys.maxsize
	best_sub_node = None
	# Travel all sub nodes to find the best one
	for sub_node in node.get_children():
		# Ignore exploration for inference
		if is_exploration:
			C = 1 / math.sqrt(2.0)
		else:
			C = 0.0
		# UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
		left = sub_node.get_quality_value() / sub_node.get_visit_times()
		right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
		score = left + C * math.sqrt(right)
		if score > best_score:
			best_sub_node = sub_node
			best_score = score
	return best_sub_node

def main():
#	total_point=[x for x in range(1,21)]
	global dic
	global setA
	global setB
	point_num = 20
	edge_num = 30
	total_edge=[(1,2),(1,5),(2,3),(3,4),(4,5),
				(1,16),(2,15),(3,13),(4,8),(5,6),
(15,17),(14,15),(13,14),(11,13),(8,11),(7,8),(6,7),(6,18),(16,18),(16,17),
				(17,20),(12,14),(10,11),(7,9),(18,19),
				(19,20),(9,19),(9,10),(10,12),(12,20)]
	for i in total_edge:
		dic[i]=0
	while graph_cut_finish(setA,setB,dic)==False:
		# print("hhh")
		next_state = MCTS()
		new_policy = next_state.get_cumulative_choices()
		print(next_state.get_cumulative_choices())
		setA = new_policy[0].copy()
		setB = new_policy[1].copy()
		dic = new_policy[2].copy()

	cut_number = 0
	for kk in dic.keys():
		if dic[kk]==1:
			cut_number +=1
	print("max cut number: {}, point set A: {}, point set B: {}".format(
		cut_number, setA, setB))
main()

# {(1, 2): 1, (1, 5): 1, (2, 3): 1, (3, 4): 0, (4, 5): 1, (1, 16): 0, (2, 15): 1, (3, 13): 1, 
# (4, 8): 1, (5, 6): 1, (15, 17): 1, (14, 15): 1, (13, 14): 0, (11, 13): 1, (8, 11): 1, (7, 8): 0, 
# (6, 7): 1, (6, 18): 1, (16, 18): 1, (16, 17): 1, (17, 20): 1, (12, 14): 1, (10, 11): 1, (7, 9): 1, 
# (18, 19): 0, (19, 20): 1, (9, 19): 1, (9, 10): 1, (10, 12): 1, (12, 20): 0}]
# max cut number: 24, point set A: [13, 17, 19, 7, 10, 8, 14, 5, 2, 18], 
# point set B: [11, 20, 9, 15, 12, 4, 16, 3, 1, 6]