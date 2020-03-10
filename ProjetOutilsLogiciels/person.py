class Person:
    
    HORIZONTAL_MINAUTORE = 16
    VERTICAL_MINAUTORE   = 32
    ARIANE               = 64
    THESEE               = 128
    PORTE                = 256
    
    def __init__(self, x, y, type, map):
        self.x = x
        self.y = y
        self.type = type
        self.map
        
    
    def can_move_up(self):
        if self.y > 0:
            return True
            
    
    def move_up(self):
        if self.y > 0:
            self.y -= 1
        return self
        
            
    def can_move_down(self):
        if self.y < self.map.get_size() :
            return True
            
    
    def move_down(self):
        if self.y < self.map.get_size() :
            self.y += 1
        return self
        
            
    def can_move_left(self):
        if self.x > 0 :
            return True
            
            
    def move_left(self):
        if self.x > 0 :
            self.x -= 1
        return self
        
            
    def can_move_right(self):
        if self.x < self.map.get_size() :
            return True
            
            
    def move_right(self):
        if self.x < self.map.get_size() :
            self.x += 1
        return self
        
        
        
    def is_in_vertical_danger(self) :
        danger = 0
        # top
        for i in range(self.y, self.map.get_size() ) :
            danger |=  ( self.tab[self.x][i] 
                             & Person.VERTICAL_MINAUTORE
                             & Person.HORIZONTAL_MINAUTORE )
                             
            if self.tab[self.x][i] | MAP.TOP_WALL :
                break
                
        # bottom
        for i in range(self.y, -1, -1) :
            danger |=  ( self.tab[self.x][i] 
                             & Person.VERTICAL_MINAUTORE
                             & Person.HORIZONTAL_MINAUTORE )
                             
            if self.tab[self.x][i] | MAP.BOTTOM_WALL :
                break
            
        return danger
        
        
    def is_in_horizontal_danger(self) :
        danger = 0
        
        # left
        for i in range(self.x, -1, -1 ) :
            danger |= ( self.tab[i][self.y]
                            & Person.HORIZONTAL_MINAUTORE
                            & Person.VERTICAL_MINAUTORE )
                            
            if self.tab[i][self.y] | MAP.LEFT_WALL :
                break
        
        # right
        for i in range(self.x, self.map.get_size() ) :
            danger |= ( self.tab[i][self.y]
                            & Person.HORIZONTAL_MINAUTORE
                            & Person.VERTICAL_MINAUTORE )
                            
            if self.tab[i][self.y] | MAP.RIGHT_WALL :
                break
        
            
    def is_in_danger(self) :
        return is_in_vertical_danger() | is_in_horizontal_danger()
            