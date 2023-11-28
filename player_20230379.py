import numpy as np
import scipy.stats
import yut.engine

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
	my_duplicates = [ sum(np == p for np in my_positions) for p in my_positions ]
	enemy_duplicates = [ sum(np == p for np in enemy_positions) for p in enemy_positions ]
	multipliers = [ 1, 1, 0.7, 0.4, 0.3 ]

	return - sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(my_positions,my_duplicates) ) \
			+ sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(enemy_positions,enemy_duplicates) ) \
			+ ( +1 if throw_again else 0 ) + calc_safe_score(my_positions, enemy_positions)*34 - calc_safe_score(enemy_positions, my_positions)*19.3

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
	for s in range(1000):
		winner = engine.play( p2, p, seed=s )
		if winner == 0:
			win_count += 1
	ans = win_count
	win_count = 0
	for s in range(1000):
		winner = engine.play( p, p2, seed=s )
		if winner == 1:
			win_count += 1
	print(ans, win_count)
	



