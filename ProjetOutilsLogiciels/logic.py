import upemtk
from draw import Draw
from map import Map


def mino_turn(mino):
    # Todo : remove code duplication
    if mino.can_move_left():
        mino.move_left()

    if mino.can_move_right():
        mino.move_right()

    if mino.can_move_up():
        mino.move_up()

    if mino.can_move_down():
        mino.move_down()


class Logic:

    def __init__(self):
        self.map = Map()
        self.map.make_map_with_file("./maps/sandbox.txt")
        self.drawer = Draw(1000, self.map.get_size())

    def start(self):
        self.drawer.start()

    def player_round(self):
        ev = upemtk.attente_clic_ou_touche()
        if ev[2] == "Touche":
            ariane = self.map.get_ariane()

            if ev[1] == "Up" and ariane.can_move_up():
                ariane.move_up()

            if ev[1] == "Down" and ariane.can_move_down():
                ariane.move_down()

            if ev[1] == "Left" and ariane.can_move_left():
                ariane.move_left()

            if ev[1] == "Right" and ariane.can_move_right():
                ariane.move_right()

    def thesee_round(self):
        thesee = self.map.get_thesee()

        if thesee.can_move_up():
            thesee.move_up()

        if thesee.can_move_down():
            thesee.move_down()

        if thesee.can_move_left():
            thesee.move_left()

        if thesee.can_move_right():
            thesee.move_right()

    def minos_v_turn(self):
        minos_v = self.map.get_v_mino()

        for mino_v in minos_v:
            mino_turn(mino_v)

    def minos_h_turn(self):
        minos_h = self.map.get_h_mino()

        for mino_h in minos_h:
            mino_turn(mino_h)

    def round(self):
        self.drawer.draw_laby(self.map, self.map.get_entities_list())
        self.player_round()
        self.thesee_round()
        self.minos_v_turn()
        self.minos_h_turn()

    def rounds(self):
        while True:
            self.round()

        self.drawer.stop()
