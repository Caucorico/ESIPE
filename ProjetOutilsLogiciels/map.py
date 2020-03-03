class Map:
    
    def __init__(self):
        self.map = []
        
    def make_map_with_file(self, filename):
        file = open(filename, "r")
        file.readline()
        lines = []
        
        for line in file:
            new_line = []
            for char in line:
                new_line.append(char)                    
            lines.append(new_line)
            
        print(lines)
            
        for i in range(len(lines)):
            new_line = []
            for j in range(len(lines[i])):
                char = lines[i][j]
                print(lines[i][j])
                if ( char == ' '
                    or char == 'A'
                    or char == 'V'
                    or char == 'H'
                    or char == 'P'
                    or char == 'T'
                    ):
                    
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
                    
                    if lines[i][j+1] == '|':
                        print('oui')
                        case |= 1
                        
                    if lines[i][j-1] == '|':
                        print('oui')
                        case |= 4
                        
                    if lines[i+1][j] == '-':
                        print('oui')
                        case |= 2
                        
                    if lines[i-1][j] == '-':
                        print('oui')
                        case |= 8
                        
                    new_line.append(case)
            self.map.append(new_line)
        
    
    def terminal_display(self):
        print(self.map)
        return;
        for line in self.map:
            for case in line:
                print(case)
        
                    