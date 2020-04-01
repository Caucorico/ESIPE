import upemtk
from draw import Draw
from map import Map


def mino_h_turn(mino_h):
    if mino_h.can_move_up():
        mino_h.move_up()

    if mino_h.can_move_down():
        mino_h.move_down()

    if mino_h.can_move_left():
        mino_h.move_left()

    if mino_h.can_move_right():
        mino_h.move_right()


def mino_v_turn(mino_v):
    if mino_v.can_move_left():
        mino_v.move_left()

    if mino_v.can_move_right():
        mino_v.move_right()

    if mino_v.can_move_up():
        mino_v.move_up()

    if mino_v.can_move_down():
        mino_v.move_down()


class Logic:
    VICTORY = 1
    IN_PROGRESS = 0
    LOOSE = -1

    def __init__(self):
        self.map = Map()
        self.map.make_map_with_file("./maps/sandbox.txt")
        self.drawer = Draw(1000, self.map.get_size())

    def start(self):
        self.drawer.start()

    def player_round(self, ev):
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
            mino_v_turn(mino_v)

    def minos_h_turn(self):
        minos_h = self.map.get_h_mino()

        for mino_h in minos_h:
            mino_h_turn(mino_h)

    def check_state(self):
        thesee = self.map.get_thesee()
        ariane = self.map.get_ariane()
        door = self.map.get_door()

        if self.map.mino_on_case(thesee.x, thesee.y):
            return -1

        if self.map.mino_on_case(ariane.x, ariane.y):
            return -1

        if door.x == thesee.x == ariane.x and door.y == thesee.y == ariane.y:
            return 1

        return 0

    def round(self):
        self.drawer.draw_laby(self.map, self.map.get_entities_list())

        ev = upemtk.attente_clic_ou_touche()
        if ev[2] == "Touche":
            self.player_round(ev)
            self.thesee_round()
            self.minos_v_turn()
            self.minos_h_turn()

    def rounds(self):

        while True:
            self.round()
            state = self.check_state()
            if state == -1:
                self.drawer.loose_screen()
                break
            elif state == 1:
                self.drawer.victory_screen()
                break

        upemtk.attente_clic_ou_touche()

        self.drawer.stop()
