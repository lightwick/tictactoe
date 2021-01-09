from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line        
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
import random

import neuralnet as nn

width, height = Window.size
myTurn = True

class action_value(GridLayout):   
    def __init__(self, **kwargs):
        super(action_value, self).__init__(**kwargs)
        self.rows = 3
        self.cols = 3
        self.width=width
        self.height=height
        self.show_values()
        
    def on_touch_down(self,touch):
        self.clear_widgets()
        self.show_values()

    def show_values(self):
        for i in range(9):
            value=""
            if nn.grid[i//3][i%3] == -1:
                value = str(round(float(nn.getAV()[0][i]),2))
            self.add_widget(Label(text=value, font_size=50))
            
class play(Widget):
    
    def __init__(self, **kwargs):
        super(play, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
    def on_touch_down(self,touch):
        # player/model
        x = int(touch.x//(width/3)+1)
        y = int(touch.y//(height/3)+1)
        nn.grid[3-y][x-1]=0
        symbol='o'
        tmp = Label(text=symbol, font_size=250, pos=(80+270*(x-1), 65+205*(y-1)))
        self.add_widget(tmp)

        print("placed")
        # cpu
        cpu_i,cpu_j=self.find_random_legal_moves()
        if cpu_i == None:
            return False
        tmp = Label(text='x', font_size=250, pos=(80+270*cpu_j, 65+205*(2-cpu_i)))
        self.add_widget(tmp)
        nn.grid[cpu_i][cpu_j]=1
        
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left':
            self.k.x -= 5
        elif keycode[1] == 'right':
            self.k.x += 5
        elif keycode[1] == 'up':
            self.k.y += 5
        elif keycode[1] == 'down':
            self.k.y -= 5
    def find_random_legal_moves(self):
        print(nn.grid)
        def full_board():
            for i in range(3):
                for j in range(3):
                    if nn.grid[i][j]==-1:
                        return False
            return True
        if(full_board()):
            print("returned none")
            return None,None
        
        while True:
            i = random.randrange(3)
            j = random.randrange(3)
            if nn.grid[i][j]==-1:
                return i,j
    
# 80 + 270 x
# 65 + 205 y
        return True

        
class TicTacToe(App):
    def build(self):
        game = Widget()
        with game.canvas:
            Line(points=[0,height/3,width,height/3], width = 3)
            Line(points=[0,height/3*2,width,height/3*2], width = 3)
            Line(points=[width/3,0, width/3 , height], width = 3)
            Line(points=[width/3*2,0, width/3*2 , height], width = 3)
        game.add_widget(play(), index=0)
        game.add_widget(action_value(),index=1)
        return game
    def on_stop(self):
        Window.close()
        
if __name__=="__main__":
    TicTacToe().run()
