import upemtk
from draw import Draw
from map import Map


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

    def mino_v_turn(self, mino_v):
        if mino_v.can_move_left():
            mino_v.move_left()

        if mino_v.can_move_right():
            mino_v.move_right()

        if mino_v.can_move_up():
            mino_v.move_up()

        if mino_v.can_move_down():
            mino_v.move_down()

    def minos_v_turn(self):
        minos_v = self.map.get_v_mino()

        for mino_v in minos_v:
            self.mino_v_turn(mino_v)

    def mino_h_turn(self, mino_h):
        # Todo : remove code duplication
        if mino_h.can_move_left():
            mino_h.move_left()

        if mino_h.can_move_right():
            mino_h.move_right()

        if mino_h.can_move_up():
            mino_h.move_up()

        if mino_h.can_move_down():
            mino_h.move_down()

    def minos_h_turn(self):
        minos_h = self.map.get_h_mino()

        for mino_h in minos_h:
            self.mino_h_turn(mino_h)

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
