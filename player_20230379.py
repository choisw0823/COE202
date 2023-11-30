import numpy as np
import scipy.stats
import yut.engine
import itertools
from functools import lru_cache

distance_to_goal = np.zeros( yut.rule.FINISHED+1 )
outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=5)
my_multiple = 0
enemy_multiple = 0
for _ in range(10):
	for s in range( yut.rule.FINISHED-1, -1, -1):
		weighted_sum = 0.0
		for outcome, prob in zip( outcomes, probs ):
			pos = s
			for ys in outcome:
				pos = yut.rule.next_position( pos, ys, True )
			weighted_sum += ( 1 + distance_to_goal[pos] ) * prob
		distance_to_goal[s] = weighted_sum

def evaluate_score( my_positions, enemy_positions, throw_again ):
	return -predict(enemy_positions, my_positions, throw_again) + (1 if throw_again else 0)

@lru_cache
def calc_total_score(my_positions, enemy_positions, throw_again):
	my_duplicates = [ sum(np == p for np in my_positions) for p in my_positions ]
	enemy_duplicates = [ sum(np == p for np in enemy_positions) for p in enemy_positions ]
	multipliers = [ 1, 1, 0.7, 0.4, 0.3 ]

	return - sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(my_positions,my_duplicates) ) \
			+ sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(enemy_positions,enemy_duplicates) ) \
			+(1 if throw_again else 0) + calc_safe_score(my_positions, enemy_positions)*6 - calc_safe_score(enemy_positions, my_positions)*3

def calc_safe_score(my_positions, enemy_positions):
	prob = yut.rule._prob_yutscores
	safe = [0]*6
	for yut_index, yut_score in enumerate([1,2,3,4,5]):
		score = prob[yut_index]
		predict_scores = [0]
		for mal in range(4):
			legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mal, yut_score, True )
			if legal_move: predict_scores.append(num_mals_caught)
		score *= min(1,max(predict_scores))
		safe[yut_index] = score
	return sum(safe)

def safe_recommend(my_positions, enemy_positions, available_yutscores, rec_count=3):
	if not available_yutscores: return [calc_safe_score(my_positions, enemy_positions),-1,-1,-1]
	if rec_count == 0: return [0,-1,-1,-1]

	min_val = [999999, -1,-1,-1]
	for yut_index, yut_score in enumerate(available_yutscores):
		for mal in range(4):
			for shortcut in [True, False]:
				legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mal, yut_score, shortcut )
				if legal_move:
					result = safe_recommend(next_my_positions, next_enemy_positions, available_yutscores[0:yut_index]+available_yutscores[yut_index+1:len(available_yutscores)], rec_count-1)
					if min_val[0]>result[0]:
						min_val = [result[0], mal, yut_score, shortcut]
	return min_val
safe_limit = 0.4

def kill_able(my_positions, enemy_positions, available_yutscores, rec_count=3):
	if rec_count==0:
		return [-1,-1,-1]
	for yut_index, yut_score in enumerate(available_yutscores):
		for mal in range(4):
			for shortcut in [True, False]:
				legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mal, yut_score, shortcut )
				if legal_move:
					if num_mals_caught:
						return [mal, yut_score, shortcut]
					if len(available_yutscores)>1:
						if kill_able(next_my_positions, next_enemy_positions, available_yutscores[0:yut_index]+available_yutscores[yut_index+1:len(available_yutscores)], rec_count-1)[0]!=-1:
							return [mal, yut_score, shortcut]
						
	return [-1,-1,-1]#false
prob = yut.rule._prob_yutscores

available_yut = []
for i,yut_score in enumerate([1,2,3,4,5,-1]):
	if yut_score == 4 or yut_score == 5:
		for j,yut_score2 in enumerate([1,2,3,4,5,-1]):
			if prob[i]*prob[j]>0.02:
				available_yut.append([[yut_score, yut_score2], prob[i]*prob[j]])
	else:
		available_yut.append([[yut_score], prob[i]])

@lru_cache
def predict(my_positions, enemy_positions, throw_again, rec_count=2):
	if rec_count == 0:
		return calc_total_score(my_positions, enemy_positions, throw_again)
	ans = 0
	for yut_info in available_yut:
		result = -999999
		for mals in itertools.product(list(range(4)), repeat=len(yut_info[0])):
			next_my_positions, next_enemy_positions = my_positions[:], enemy_positions[:]
			illegal = False
			for i,mal in enumerate(mals):
				legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( next_my_positions, next_enemy_positions, mal, yut_info[0][i], True )
				if not legal_move: 
					illegal = True
					break
			if illegal: continue
			pre_result = -predict(tuple(sorted(list(next_enemy_positions))), tuple(sorted(list(next_my_positions))), num_mals_caught>0, rec_count-1)
			result = max(pre_result, result)
		ans += result*yut_info[1]
		
	return ans+(1 if throw_again else 0)

class MyPlayer(yut.engine.Player):
	def name(self):
		return "Example"

	def action(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state
		scores = []
		kill_move = kill_able(my_positions, enemy_positions, available_yutscores)
		# if safe_score>safe_limit:
		# 	#생존율을 높이는 방향으로 움직임
		# 	return safe_recommend(my_positions, enemy_positions, available_yutscores)
		if kill_move[0] != -1:
			#죽일 수 있는 말을 잡음
			return kill_move + [""]
		for mi, mp in enumerate(my_positions):
			if mp == yut.rule.FINISHED:
				continue
			for ys in available_yutscores:
				for shortcut in [True, False]:
					legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mi, ys, shortcut )
					if legal_move:
						scores.append( (evaluate_score(next_my_positions, next_enemy_positions, num_mals_caught>0), mi, ys, shortcut ) )
		scores.sort(reverse=True)
		return scores[0][1], scores[0][2], scores[0][3], ""
	
if __name__ == "__main__":
	import example_player
	p = example_player.ExamplePlayer()
	p2 = MyPlayer()
	engine = yut.engine.GameEngine()
	ans = 0
	win_count = 0
	for s in range(100):
		winner = engine.play( p2, p)
		if winner == 0:
			print('선공 승!')
			win_count += 1
		else:
			print('패배!')
	ans = win_count
	print(ans)
	win_count = 0
	for s in range(100):
		winner = engine.play( p, p2 )
		if winner == 1:
			print('후공 승!')
			win_count += 1
		else:
			print('패배!')
	print(ans, win_count)
	



