# -*- coding: utf-8 -*-
"""
迷路の世界

・迷路はエクセルファイルで設計する
　迷路の範囲は
　シートの左上（A1）から
　EOG（End of Grid）があるセルの左上まで
　とする（EOGの先にあるデータは無視する）
 
・セルの種類
　開始点をS（Start）、終点をG（Goal）、
　通れないセルをW（Wall）とする
　トラップセルT（Trap）を踏むと死亡する（報酬-100） 
 数値のセルはその値だけ報酬が得られることを表す（1以上9以下、1は空白と同値）
 空白セルは報酬1

・以下の場合はエラー
　‐EOGが無い、または複数ある
　‐EOGがA列または1行目にある（範囲を生成できない）
　‐SまたはGが無い、またはGが複数ある
 
"""

import pandas as pd

class GridWorld:

    """ グリッドの迷路クラス """    
    def __init__(self, filename):
        self.__gw_df = pd.read_excel(filename, header=None, index_col=None)
        self.__gw = pd.DataFrame() # 迷路の範囲
        self.__S_pos = () # 始点の座標
        self.__G_pos = () # ゴールの座標
        self.__action_list = ['U', 'D', 'L', 'R'] # この世界で可能な行動
        self.__state_to_pos = {} # ステートをキーに、座標をバリューにした辞書
        self.__pos_to_state = {} # 座標をキーに、ステートをバリューにした辞書
                        
    def create_grid_world(self):
        """ 迷路の生成 """
        try:
            # 有効な迷路の範囲を取得
            assert self.is_griddata_valid(), 'grid data is invalid'
            # NaNをF（Floor）に塗り替える
            self.__gw = self.__gw.fillna('1')

            # ステートidを振っていく
            self.__start_s_id = ''
            self.__goal_s_id = ''
            s_idx = 0
            for c in range(len(self.__gw.columns)):
                for r in range(len(self.__gw)):   
                    cell_attr = self.get_cell_attribute(r, c)
                    if cell_attr != 'W':
                        s_idx += 1
                        s_id = f's{s_idx}'
                        self.__state_to_pos[s_id] = (r, c)
                        self.__pos_to_state[(r, c)] = s_id
                        
                        if cell_attr == 'S': # スタート地点なら初期値として記憶
                            self.__start_s_id = s_id
                        if cell_attr == 'G': # ゴール地点も記憶
                            self.__goal_s_id = s_id 
            
            # 初期ステートはSの位置
            self.__state = self.__start_s_id
            
            print('===== Grid Maze World =====')
            print(self.__gw)
            print('===========================')
            
        except AssertionError as err:
            print(f'AssertionError: {err}')
            
        
    def is_griddata_valid(self):
        """ ロードしたグリッドexcelデータが有効か判断する処理 """

        # EOGの存在確認と座標、重複チェック
        EOG_pos, is_EOG_valid = self.get_pos_of_target_in_df(self.__gw_df, 'EOG')
        # EOG有効性チェック
        if is_EOG_valid is False:
            return False        
        if EOG_pos[0] == 0 or EOG_pos[1] == 0:
            return False
        
        # 有効な領域（迷路の範囲）の切り出し
        self.__gw = self.__gw_df.iloc[ :EOG_pos[0], :EOG_pos[1] ]        

        # S, Gの取得、有効性チェック
        self.__S_pos, is_S_valid = self.get_pos_of_target_in_df(self.__gw, 'S')
        self.__G_pos, is_G_valid = self.get_pos_of_target_in_df(self.__gw, 'G')        
        if is_S_valid is False or is_G_valid is False:
            return False
        
        return True
    
    def get_pos_of_target_in_df(self, df, target):
        """ DataFlameに含まれるtargetの座標を取得する """
        """ 存在しないor重複が見つかったらFalseを返す """        
        pos = ()
        if target not in df.values:
            return pos, False

        for col_header in df.columns:
            col_idx = df.columns[col_header]
            for row_idx in range(len(df)):         
                val = str(df[col_header].iloc[row_idx])                
                if val == target:
                    if len(pos) == 0:
                        pos = (row_idx, col_idx)
                    else :
                        return (), False
        return pos, True
    
    
    def reset_state(self):
        """ ステートをスタート地点にリセットする """
        self.__state = self.__start_s_id
        return    
    
    def get_grid_size(self):
        """ 迷路のサイズを返却する """
        return len(self.__gw), len(self.__gw.columns)        
    
    def get_cell_attribute(self, row, col):
        """ row行 col列のセルの属性（'S'とか'G'とか）を取得する """
        return self.__gw.iloc[row, col]
    
    def get_action_list(self):
        """ エージェントが取れるアクション一覧を返す """
        return self.__action_list
    
    def get_state_list(self):
        """ エージェントにステート一覧を返す """
        return self.__state_to_pos.keys()
    
    def get_state(self):
        """ 今どの状態かを返す """
        return self.__state
    
    def get_position(self):
        """ 今どの座標にいるかを返す """
        return self.__state_to_pos[self.__state]

    def update_state(self, state):
        """ 状態の更新 """
        if state in self.__state_to_pos.keys():
            self.__state = state
        return
    
    def state_transition(self, action):
        """ actionをエージェントから受け取って報酬とステートを返す処理 """

        """ 現在の状態から行動可能か判断 """
        """ デフォルトを遷移失敗時のreward, stateとし、遷移成功なら更新する """
        """ 遷移に成功しても、そこがトラップなら即死でブレークする """
        
        cur_pos = self.__state_to_pos[self.__state]        
        reward = -1
        new_state = self.__state
        
        if action == 'U':
            # 上が迷路の上辺でない、かつ壁でないなら遷移成功
            if cur_pos[0] > 0:
                upper_attribute = self.get_cell_attribute(cur_pos[0]-1, cur_pos[1])
                if upper_attribute != 'W':
                    new_state = self.__pos_to_state[(cur_pos[0]-1, cur_pos[1])]
                    reward = 1

        elif action == 'D':
            # 下が迷路の下辺でない、かつ壁でないなら遷移成功
            if cur_pos[0] < len(self.__gw)-1:
                down_attribute = self.get_cell_attribute(cur_pos[0]+1, cur_pos[1])
                if down_attribute != 'W':
                    new_state = self.__pos_to_state[(cur_pos[0]+1, cur_pos[1])]
                    reward = 1

        elif action == 'L':
            # 左が迷路の左辺でない、かつ壁でないなら遷移成功
            if cur_pos[1] > 0:
                left_attribute = self.get_cell_attribute(cur_pos[0], cur_pos[1]-1)
                if left_attribute != 'W':
                    new_state = self.__pos_to_state[(cur_pos[0], cur_pos[1]-1)]
                    reward = 1
                    
        else: # 'R'
            if cur_pos[1] < len(self.__gw.columns)-1:
                right_attribute = self.get_cell_attribute(cur_pos[0], cur_pos[1]+1)
                if right_attribute != 'W':
                    new_state = self.__pos_to_state[(cur_pos[0], cur_pos[1]+1)]
                    reward = 1
                    
        # reward 1 は遷移成功を表すので、その時の遷移先セルの情報をチェック
        if reward == 1:
            # セル属性を見るために遷移先（新ステート）の座標を取得
            new_pos = self.__state_to_pos[new_state]           
            new_cell_attr = self.get_cell_attribute(new_pos[0], new_pos[1])
            
            if new_cell_attr == 'G':
                """ ゴールなら報酬弾むよ """
                reward = 100                
            elif new_cell_attr == 'T':
                """ トラップは死ね """
                reward = -100                
            elif new_cell_attr != 'S':
                """ 数値セル（上記if文の後ではS以外） """
                reward = int(new_cell_attr)

        return reward, new_state
    
    
    def is_arrived_at_goal(self):
        """ ゴールについたか判定 """
        if self.__state == self.__goal_s_id:
            return True
        else:
            return False
        
    