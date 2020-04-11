# -*- coding: utf-8 -*-
"""

簡単な強化学習タスクでは
・世界はハードコーディングする（閉じた世界）
 - 報酬の値
 - 壁とかトラップとか行動可能か否か

@author: Katsuki
"""

import grid_world
import RL_agent
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

#%%
def solve_maze(gw:grid_world.GridWorld, 
               agent:RL_agent.ReinforcementLearningAgent,
               step):
    # 迷路のサイズを取得
    row_size, col_size = gw.get_grid_size()
    
    # 描画領域
    fig = plt.figure(figsize=(col_size, row_size))    
    plt.xlim(-1, col_size)
    plt.ylim(row_size, -1)
    
    # マップ表示
    for r in range(row_size):
        for c in range(col_size):
            attr = gw.get_cell_attribute(r, c)
            if attr == 'W':
                plt.plot(c, r, marker='s', markersize=46, color='black')
            elif attr == 'S':
                plt.plot(c, r, marker='s', markersize=40, color='b')
            elif attr == 'G':
                plt.plot(c, r, marker='s', markersize=40, color='r')
    
    plt.show()
    
    """ 初期の描画 """
    gw.reset_state() 
    y, x = gw.get_position()
    plt.plot(x, y, marker='o', markersize=20)
    plt.title('初期状態')
    plt.pause(1)
    
    """ step回数までの動きをプロットしてみる """
    state = gw.get_state()
    action = agent.select_action_on_policy(state, 1)    

    for t in range(step):
        reward, new_state = gw.state_transition(action)
        new_action = agent.select_action_on_policy(new_state, epi)
        state = gw.get_state()
        agent.update_Q_table_by_SARSA(st=state, act=action, rwd=reward, st_n=new_state, act_n=new_action)
        gw.update_state(new_state)           
        action = new_action        

        """ 結果をプロット """
        plt.cla()        
        plt.xlim(-1, col_size)
        plt.ylim(row_size, -1)
        for r in range(row_size):
            for c in range(col_size):
                attr = gw.get_cell_attribute(r, c)
                if attr == 'W':
                    plt.plot(c, r, marker='s', markersize=46, color='black')
                elif attr == 'S':
                    plt.plot(c, r, marker='s', markersize=40, color='b')
                elif attr == 'G':
                    plt.plot(c, r, marker='s', markersize=40, color='r')
                elif attr == 'T':
                    plt.plot(c, r, marker='s', markersize=46, color='gray')
            
        y, x = gw.get_position()
        plt.plot(x, y, marker='o', markersize=20)
        plt.title(f'{t}回目')
        plt.pause(0.5)
        
        """ ゴールなら終了 """
        if gw.is_arrived_at_goal() is True:
            plt.title(f'{t}回目 CLEAR!')
            print('CLEAR!')
            break
        
        """ トラップに嵌れば死亡 """
        if gw.is_into_trap() is True:
            plt.title(f'{t}回目 TRAPPED!')
            print('YOU DIED!')
            break   
        

#%%
if __name__ == '__main__':
    
    """ 迷路（環境）生成 """
    filename = '../data/grid_world_cliff.xlsx'
    gw = grid_world.GridWorld(filename)
    gw.create_grid_world()
    
    """ エージェント生成 """
    agent = RL_agent.ReinforcementLearningAgent(
            action_list = gw.get_action_list(),
            state_list = gw.get_state_list() )
        
    """ 学習パラメータ """
    step = 150
    episode = 7000
    
    """ 学習実行 """
    print('SARSA法')
    
    g_flg = False
    
    for epi in range(episode):
        # リセット
        gw.reset_state()        
        # スタート地点の状態s0を取得
        state = gw.get_state()
        # 行為a0を状態s0での政策piから選択
        action = agent.select_action_on_policy(state, epi)      

        if epi%100 == 0:
            print(f'---------- episode:{epi} -------------')  
            
        for t in range(step):
            # 行為atを実行、報酬rt+1, 状態st+1を取得
            reward, new_state = gw.state_transition(action)
            # 行為at+1を状態st+1での政策piから選択
            new_action = agent.select_action_on_policy(new_state, epi)
            
            #if epoc%10000 == 0 and t%10 == 0:
            #    print(f'   ======= step:{t} =======')  

            # Q(st,at)を更新
            state = gw.get_state()
            agent.update_Q_table_by_SARSA(st=state, act=action, rwd=reward, st_n=new_state, act_n=new_action)

            # st <- st+1, at <- at+1
            gw.update_state(new_state)           
            action = new_action

            # ゴールに到達していたらbreak
            if gw.is_arrived_at_goal() is True:
                print(f'Goal! at episode {epi}')
                break
            
            # トラップでも終了
            if gw.is_into_trap() is True:
                print(f'!! Trapped !! at episode {epi}')
                break            
    
    """ 学習終了後のQtable """        
    agent.show_Q_table()
    
    """ 実際にたどってみる """
    solve_maze(gw=gw, agent=agent, step=step)
    
    
            