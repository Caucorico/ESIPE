from __future__ import annotations
from copy import deepcopy
import logic
import time


class Solver:

    def __init__(self, logic_obj, drawer):
        self.logic = logic_obj
        self.visited_states = set()
        self.drawer = drawer
        self.solution = []

    @staticmethod
    def from_logic(logic, drawer):
        new_logic = deepcopy(logic)
        new_first_state = new_logic.history.pop()
        new_logic.history = [new_first_state]
        return Solver(new_logic, drawer)

    def backtracking_check_validity(self, visual=False):
        state = self.logic.check_state()
        if state == logic.Logic.VICTORY:
            return True
        elif state == logic.Logic.LOOSE:
            return False

        if visual:
            self.drawer.draw_laby(self.logic.map, self.logic.map.get_entities_list())

        self.visited_states.add(deepcopy(self.logic.map.get_state()))
        for direction in ("Up", "Down", "Left", "Right"):

            if not self.logic.map.get_ariane().can_move_to(direction):
                continue

            self.logic.add_state_to_history(deepcopy(self.logic.map.get_state()))
            self.solution.append(direction)
            self.logic.map.get_ariane().move_to(direction)
            self.logic.thesee_round()
            self.logic.minos_v_turn()
            self.logic.minos_h_turn()

            if self.logic.map.get_state() not in self.visited_states:
                if self.backtracking_check_validity(visual):
                    return True

            self.logic.return_to_last_state()
            self.solution.pop()

        return False

    def backtracking_solution(self):
        return self.solution

    def display_solution(self):
        self.reset_state()
        for direction in self.solution:
            self.logic.extern_round(direction)
            time.sleep(0.5)

    def reset_state(self):
        first_state = self.logic.history[0]
        self.logic.history = []
        self.logic.add_state_to_history(first_state)
        self.logic.map.set_state(first_state)
