from map import *

def main():
    map = Map()
    map.make_map_with_file("./maps/labyrinthe1.txt")
    map.terminal_display()
    

main()