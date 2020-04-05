# -*- coding: utf-8 -*-
"""
強化学習エージェントクラス

"""

import numpy as np

class ReinforcementLearningAgent:
    """ 強化学習のエージェントクラス """
    epsilon = 0.25 
    alpha = 0.8 # 学習率
    gamma = 0.9 # 割引率
    
    def __init__(self, action_list=[], state_list=[]):
        # 取れる行動の選択肢
        self.__a_lst = action_list
        # 全状態リスト
        self.__s_lst = state_list
        # Q関数（Qtable）
        self.__q_table = {s:{a: 0 for a in action_list} for s in state_list}
        
        
    def select_action_on_policy(self, state, episode):
        """ 政策piに従って行動を選択する """
        """ ε-greedy探索を用いる """
        eps = np.random.uniform(0, 1)
        a_idx = 0
        if 1-eps > self.epsilon: # 1-εで最適な行動
            q_val_arr = list(self.__q_table[state].values())
            a_idx = np.argmax(q_val_arr)
        else: # εでランダムに選択
            a_idx = np.random.randint(4)

        return self.__a_lst[a_idx]
    
    
    def update_Q_table_by_SARSA(self, st, act, rwd, st_n, act_n ):
        """ SARSA法でQtableを更新する """
        """ Q(st,at) <- (1-α)Q(st,at) + α(rt+1 + γQ(st+1, at+1)) """
        # 現在のQ値を取得
        q_curr = self.__q_table[st][act]
        # 1ステップ先のQ値を取得
        q_new = self.__q_table[st_n][act_n]
        # 更新
        self.__q_table[st][act] = (1-self.alpha)*q_curr + self.alpha*(rwd + self.gamma*q_new)
        
        
    def show_Q_table(self):
        """ Qtableの内部を見せる """
        a_header = 'action'
        for a in self.__a_lst:
            a_header += f'\t{a}'
        print(a_header)
        
        for s in self.__s_lst:
            s_row = f'{s}'
            for a in self.__a_lst:
                s_row += '\t{:.4g}'.format(self.__q_table[s][a])
            print(s_row)
        