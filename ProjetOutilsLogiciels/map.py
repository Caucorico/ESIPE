class Map:
    
    LEFT_WALL   = 1
    TOP_WALL    = 2
    RIGHT_WALL  = 4
    BOTTOM_WALL = 8
    
    def __init__(self):
        self.map = []
        self.size = 0
        
    def make_map_with_file(self, filename):
        file = open(filename, "r")
        self.size = int(file.readline())
        lines = []
        
        for line in file:
            new_line = []
            for char in line:
                new_line.append(char)                    
            lines.append(new_line)
            
            
        for i in range(len(lines)):
            new_line = []
            if i%2:
                for j in range(len(lines[i])):
                    if j%2 and lines[i][j] != '\n':
                        char = lines[i][j]
                        print (char) 
                        case = 0
                    
                        # |case = 1
                        # 
                        # ----
                        # case  = 2
                        #
                        # case| = 4
                        #
                        # case  = 8
                        # ----
                        
                        if lines[i][j-1] == '|':
                            case |= Map.LEFT_WALL
                            
                        if lines[i][j+1] == '|':
                            case |= Map.RIGHT_WALL
                            
                        if lines[i-1][j] == '-':
                            case |= Map.TOP_WALL
                            
                        if lines[i+1][j] == '-':
                            case |= Map.BOTTOM_WALL
                            
                        new_line.append(case)
                    
                self.map.append(new_line)
        
    
    def terminal_display(self):
        print("size="+str(self.size))
        for line in self.map:
            print(line)
            
            
    def get_size():
        return self.size
            
        
                    