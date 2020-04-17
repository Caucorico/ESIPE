from __future__ import annotations

import upemtk
from draw import Draw
from map import Map
from state import State
from copy import deepcopy
from solver import Solver


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
        self.history = []

    def start(self):
        # Save first state
        self.add_state_to_history(deepcopy(self.map.get_state()))
        self.drawer.start()

    def add_state_to_history(self, state: State):
        self.history.append(state)

    def pop_state_from_history(self) -> object:
        """
        This method pop the most recent state from the history.
        The method return a tuple. In this tuple, we can find :
            In first position : a boolean to True if the pop succeed, false otherwise.
            In second position : The State object if the pop succeed.

        :rtype: tuple
        """
        if len(self.history) < 1:
            return False, None

        return True, self.history.pop()

    def return_to_last_state(self):
        (succeed, old_state) = self.pop_state_from_history()
        if succeed:
            self.map.set_state(old_state)

    def player_round(self, ev) -> bool:

        if ev[2] == "Touche":
            ariane = self.map.get_ariane()

            if ev[1] == "Up" and ariane.can_move_up():
                # Save new state in the history
                self.add_state_to_history(deepcopy(self.map.get_state()))
                ariane.move_up()
                return True

            if ev[1] == "Down" and ariane.can_move_down():
                # Save new state in the history
                self.add_state_to_history(deepcopy(self.map.get_state()))
                ariane.move_down()
                return True

            if ev[1] == "Left" and ariane.can_move_left():
                # Save new state in the history
                self.add_state_to_history(deepcopy(self.map.get_state()))
                ariane.move_left()
                return True

            if ev[1] == "Right" and ariane.can_move_right():
                # Save new state in the history
                self.add_state_to_history(deepcopy(self.map.get_state()))
                ariane.move_right()
                return True

            if ev[1] == "r":
                self.return_to_last_state()

            if ev[1] == "c":
                solver = Solver.from_logic(self, self.drawer)
                if solver.backtracking_check_validity():
                    self.drawer.display_chance(True)
                    print("Configuration gagnante !")
                else:
                    self.drawer.display_chance(False)
                    print("Configutation perdante !")

            if ev[1] == "v":
                print("Visual")
                solver = Solver.from_logic(self, self.drawer)
                solver.backtracking_check_validity(True)

            if ev[1] == "s":
                print("Solution")
                solver = Solver.from_logic(self, self.drawer)
                if solver.backtracking_check_validity():
                    print(solver.backtracking_solution())
                else:
                    self.drawer.display_chance(False)

            if ev[1] == "d":
                print("Visual Solution")
                solver = Solver.from_logic(self, self.drawer)
                if solver.backtracking_check_validity():
                    print(solver.backtracking_solution())
                    solver.display_solution()
                else:
                    self.drawer.display_chance(False)

        return False

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

    def extern_round(self, direction: str) -> None:
        self.map.get_ariane().move_to(direction)
        self.thesee_round()
        self.minos_v_turn()
        self.minos_h_turn()
        self.drawer.draw_laby(self.map, self.map.get_entities_list())

    def round(self):
        self.drawer.draw_laby(self.map, self.map.get_entities_list())
        ev = upemtk.attente_clic_ou_touche()

        # If the history is empty, add the current position to it !
        if len(self.history) == 0:
            self.add_state_to_history(deepcopy(self.map.get_state()))

        if not self.player_round(ev):
            return

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
