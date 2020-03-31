from map import *
from draw import *


def main():
    map = Map()
    map.make_map_with_file("./maps/labyrinthe1.txt")

    drawer = Draw(1000, map.get_size())
    drawer.start()
    drawer.draw_laby(map, map.get_entities_list())
    drawer.stop()


main()
