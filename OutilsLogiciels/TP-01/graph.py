from upemtk import *

class Graph:    
    def __init__(self, size_x, size_y, origin_x, origin_y, listes, colors):
        self.size_x = size_x
        self.size_y = size_y
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.listes = listes
        self.colors = colors
        self.current_line = 0
    
    def draw_x(self):
        ligne(0, self.origin_y, self.size_x, self.origin_y)
        
    def draw_y(self):
        ligne(self.origin_x, 0, self.origin_x, self.size_y)
        
    def draw_line(self, liste):
        for i in range(0, len(liste)-1):
            ligne(
                self.origin_x + liste[i][0],
                self.origin_y - liste[i][1],
                self.origin_x + liste[i+1][0],
                self.origin_y - liste[i+1][1],
                self.colors[self.current_line]
            )
            
        self.current_line+=1
            
    def draw_all_lines(self):
        for liste in self.listes:
            self.draw_line(liste)
            
    def draw_graph(self):
        cree_fenetre(self.size_x, self.size_y)
        self. draw_x()
        self.draw_y()
        self.draw_all_lines()
        mise_a_jour()
        attente_clic_ou_touche()
        fermer_fenetre()