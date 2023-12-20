import numpy as np
import yut.engine
import itertools
from functools import cache
import yut.rule
import example_player
import time 

distance_to_goal = np.zeros(yut.rule.FINISHED + 1)
outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=5)
my_multiple = 0
enemy_multiple = 0

for _ in range(10):
    for s in range(yut.rule.FINISHED - 1, -1, -1):
        weighted_sum = 0.0
        for outcome, prob in zip(outcomes, probs):
            pos = s
            for ys in outcome:
                pos = yut.rule.next_position(pos, ys, True)
            weighted_sum += (1 + distance_to_goal[pos]) * prob
        distance_to_goal[s] = weighted_sum


def evaluate_score(my_positions, enemy_positions, more=[]):
    if len(more)==0:
        return -predict(enemy_positions, my_positions)
    else:
        return predict(my_positions, enemy_positions, rec_count=3, plus=tuple(more), first=True)
    	

def shortcut_score(my_positions):
    score = 0
    for loc in my_positions:
        if loc in [15]:
            score += 3
        if loc in [13,10,5,14,23,24]:
            score += 2
        if loc in [18,19]:
            score -= 1

    return score

@cache
def calc_total_score(my_positions, enemy_positions):                                #calculate total score for check how this state is good
    my_duplicates = [sum(np == p for np in my_positions) for p in my_positions]
    enemy_duplicates = [sum(np == p for np in enemy_positions) for p in enemy_positions]
    multipliers = [1, 1, 0.7, 0.4, 0.3]
                                                                                    # these codes are from professor's evaluate_score function
    return -sum(distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p, np in zip(my_positions, my_duplicates)) \
        + sum(distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p, np in zip(enemy_positions, enemy_duplicates)) \
        + (calc_safe_score(my_positions, enemy_positions) * 2                       # calculate how aggresive this state is
            - calc_safe_score(enemy_positions, my_positions) * 8                 # calculate how deffensive this state is
              + shortcut_score(my_positions))                                       # if bot can use shortcut, bot can get extra score

def calc_safe_score(my_positions, enemy_positions):
    outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=2)
    safe = 0
    for prob, yut_list in zip(probs, outcomes):
        prev_list = []
        for mal in range(4):
            if (my_positions[mal], yut_list[0]) in prev_list:
                continue
            legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mal, yut_list[0], True)
            prev_list.append((my_positions[mal], yut_list[0]))
            if legal_move:
                if len(yut_list) > 1:
                    prev_list2 = []
                    for mal in range(4):
                        if (next_my_positions[mal], yut_list[1]) in prev_list:
                            continue
                        legal_move, nnext_my_positions, nnext_enemy_positions, num_mals_caught = yut.rule.make_move(next_my_positions, next_enemy_positions, mal, yut_list[1], True)
                        prev_list2.append((next_my_positions[mal], yut_list[1]))
                        if legal_move:
                            safe += num_mals_caught * prob 
                else:
                    safe += num_mals_caught * prob
    return safe
def kill_able(my_positions, enemy_positions, available_yutscores, recur=False):
    cases = []
    from itertools import permutations
    for yut_score in available_yutscores:
        for mal in range(4):
            for shortcut in [True, False]:
                legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mal, yut_score, shortcut)
                if legal_move:
                    if num_mals_caught != 0:
                        cases.append((mal, yut_score, shortcut, next_my_positions, next_enemy_positions))

    if len(cases) != 0:
        best = (-100, [])
        for case in cases:
            score = predict(case[3], case[4], rec_count=0)
            if score > best[0]:
                best = (score, case)
        if recur:
            return best[0]
        else:
            if best[0] > -100:
                return [best[1][0], best[1][1], best[1][2]]

    if len(available_yutscores) > 1:
        best = (-100, [])
        for yut_list in list(set(permutations(available_yutscores, 2))):
            for mal in range(4):
                for shortcut in [True, False]:
                    legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mal, yut_list[0], shortcut)
                    if legal_move:
                        result = kill_able(next_my_positions, next_enemy_positions, yut_list[1:], True)
                        if result is not None:
                            if result > best[0]:
                                best = (result, [mal, yut_list[0], shortcut])
        if best[0] > -100:
            return best[1]

    return None

prob = yut.rule._prob_yutscores
available_yut = []
for i, yut_score in enumerate([1, 2, 3, 4, 5, -1]):
    if yut_score == 4 or yut_score == 5:
        for j, yut_score2 in enumerate([1, 2, 3, 4, 5, -1]):
            if prob[i] * prob[j] > 0.01:
                available_yut.append([[yut_score, yut_score2], prob[i] * prob[j]])
    else:
        available_yut.append([[yut_score], prob[i]])

@cache
def predict(my_positions, enemy_positions, rec_count=2, plus=None, first=False):
    my_positions = tuple(my_positions)
    enemy_positions = tuple(enemy_positions)
    if rec_count == 0:
        return calc_total_score(my_positions, enemy_positions)
    ans = 0
    if plus is not None and len(plus) > 0 and first==False:
        available = [(list(yut) + list(plus), prob) for yut, prob in available_yut]
    elif plus is not None and len(plus) > 0 and first == True:
        available = [(list(plus), 1)]
    else:
        available = available_yut
    for yut_list, prob in available:
        result = -999999
        for yut_info in list(set(itertools.permutations(yut_list, len(yut_list)))):
            for mals in itertools.product(list(range(4)), repeat=len(yut_info)):
                next_my_positions, next_enemy_positions = my_positions[:], enemy_positions[:]
                illegal = False
                caught = False
                for i, mal in enumerate(mals):
                    legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(next_my_positions, next_enemy_positions, mal, yut_info[i], True)
                    if not legal_move:
                        illegal = True
                        break
                    if num_mals_caught > 0 and yut.rule.needs_throw_again(yut_info[i]) == False:
                        caught = True
                        pre_result = predict(tuple(sorted(list(next_my_positions))), tuple(sorted(list(next_enemy_positions))), rec_count=rec_count - 1, plus=yut_info[i + 1:])
                        break

                if illegal: continue
                if caught == False:
                    pre_result = -predict(tuple(sorted(list(next_enemy_positions))), tuple(sorted(list(next_my_positions))), rec_count=rec_count - 1)

                result = max(pre_result, result)
        ans += result * prob
    return ans

class MyAlgo(yut.engine.Player):
    def name(self):
        return "AlphaYut"

    def action(self, state):
        turn, my_positions, enemy_positions, available_yutscores = state
        scores = []
        kill_move = kill_able(my_positions, enemy_positions, available_yutscores)
        if kill_move is not None:
            return kill_move + [""]
        for mi, mp in enumerate(my_positions):
            if mp == yut.rule.FINISHED:
                continue
            for i, ys in enumerate(available_yutscores):
                for shortcut in [True, False]:
                    legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mi, ys, shortcut)
                    if shortcut == False and prev_next_m == next_my_positions and prev_next_e == next_enemy_positions:
                        break
                    else:
                        prev_next_m, prev_next_e = next_my_positions, next_enemy_positions                    
                    if legal_move:
                        if len(available_yutscores) == 1:
                            scores.append((evaluate_score(next_my_positions, next_enemy_positions, more=[]), mi, ys, shortcut))
                            #print(next_my_positions, next_enemy_positions, new_avail, scores[-1])
                        else:
                            new_avail = available_yutscores[:]
                            new_avail.remove(ys)
                            scores.append((evaluate_score(next_my_positions, next_enemy_positions, more=new_avail), mi, ys, shortcut))

        scores.sort(reverse=True)
        return scores[0][1], scores[0][2], scores[0][3], ""

if __name__ == "__main__":
    p = example_player.ExamplePlayer()
    p2 = MyAlgo()
    engine = yut.engine.GameEngine()
    ans = 0
    win_count = 0
    t = time.time()
    for s in range(100):
        event_logger = yut.engine.EventLogger()
        winner = engine.play(p2, p, game_event_listener=event_logger)
        event_logger.save( f"{t}_game_1_{s}.log" )
        if winner == 0:
            win_count += 1
        print(win_count, s+1 - win_count)

    ans = win_count
    print(ans)

    win_count = 0
    for s in range(100):
        event_logger = yut.engine.EventLogger()
        winner = engine.play(p, p2, game_event_listener=event_logger)
        event_logger.save( f"{t}_game_2_{s}.log" )
        if winner == 1:
            win_count += 1
        print(win_count, s+1 - win_count)
    print(ans, win_count)
