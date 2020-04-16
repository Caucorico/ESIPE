from __future__ import annotations
from copy import deepcopy
import logic


class Solver:

    def __init__(self, logic_obj):
        self.logic = logic_obj
        self.visited_states = set()

    @staticmethod
    def from_logic(logic):
        return Solver(deepcopy(logic))

    def backtracking_check_validity(self):
        state = self.logic.check_state()
        if state == logic.Logic.VICTORY:
            return True
        elif state == logic.Logic.LOOSE:
            return False

        self.visited_states.add(deepcopy(self.logic.map.get_state()))
        for direction in ("Up", "Down", "Left", "Right"):

            if not self.logic.map.get_ariane().can_move_to(direction):
                continue

            self.logic.add_state_to_history(deepcopy(self.logic.map.get_state()))
            self.logic.map.get_ariane().move_to(direction)
            self.logic.thesee_round()
            self.logic.minos_v_turn()
            self.logic.minos_h_turn()

            if self.logic.map.get_state() not in self.visited_states:
                if self.backtracking_check_validity():
                    return True

            self.logic.return_to_last_state()

        return False
