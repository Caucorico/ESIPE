from __future__ import annotations

from person import Person
from case import Case
from state import State


class Map:

    def __init__(self):
        self.map = []
        self.state = State()
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

        self.map = []
        for i in range(0, self.size):
            self.map.append([])
            for j in range(0, self.size):
                self.map[i].append(None)

        for line in enumerate(lines):
            if line[0] % 2:
                for char in enumerate(line[1]):
                    if char[0] % 2:
                        y = line[0] // 2
                        x = char[0] // 2

                        if lines[line[0]][char[0]] == '\n':
                            continue

                        case = Case(x, y)

                        if lines[line[0]][char[0] - 1] == '|':
                            case.set_left_wall(True)

                        if lines[line[0]][char[0] + 1] == '|':
                            case.set_right_wall(True)

                        if lines[line[0] - 1][char[0]] == '-':
                            case.set_top_wall(True)

                        if lines[line[0] + 1][char[0]] == '-':
                            case.set_bottom_wall(True)

                        if lines[line[0]][char[0]] == 'A':
                            self.state.set_ariane(Person(x, y, Person.ARIANE, self))

                        if lines[line[0]][char[0]] == 'T':
                            self.state.set_thesee(Person(x, y, Person.THESEE, self))

                        if lines[line[0]][char[0]] == 'P':
                            self.state.set_door(Person(x, y, Person.PORTE, self))

                        if lines[line[0]][char[0]] == 'H':
                            self.state.add_mino_h(Person(x, y, Person.HORIZONTAL_MINAUTORE, self))

                        if lines[line[0]][char[0]] == 'V':
                            self.state.add_mino_v(Person(x, y, Person.VERTICAL_MINAUTORE, self))

                        self.map[x][y] = case

    def case_have_top_wall(self, i, j):
        return self.map[i][j].has_top_wall()

    def case_have_bottom_wall(self, i, j):
        return self.map[i][j].has_bottom_wall()

    def case_have_left_wall(self, i, j):
        return self.map[i][j].has_left_wall()

    def case_have_right_wall(self, i, j):
        return self.map[i][j].has_right_wall()

    def get_size(self):
        return self.size

    def get_entities_list(self):
        entity_list = [self.state.get_ariane(), self.state.get_thesee(), self.state.get_door()]
        entity_list.extend(self.state.get_minos_h())
        entity_list.extend(self.state.get_minos_v())
        return entity_list

    def get_thesee(self):
        return self.state.get_thesee()

    def get_ariane(self):
        return self.state.get_ariane()

    def get_v_mino(self):
        return self.state.get_minos_v()

    def get_h_mino(self):
        return self.state.get_minos_h()

    def get_door(self):
        return self.state.get_door()

    def mino_on_case(self, x, y):
        minos_v = self.get_v_mino().copy()
        minos_h = self.get_h_mino().copy()
        minos_v.extend(minos_h)

        for mino in minos_v:
            if mino.x == x and mino.y == y:
                return True

        return False

    def get_state(self) -> State:
        return self.state

    def set_state(self, state: State) -> Map:
        self.state = state
        return self
