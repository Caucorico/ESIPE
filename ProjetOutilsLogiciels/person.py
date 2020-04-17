class Person:
    HORIZONTAL_MINAUTORE = 16
    VERTICAL_MINAUTORE = 32
    ARIANE = 64
    THESEE = 128
    PORTE = 256

    def __init__(self, x, y, type, map):
        self.x = x
        self.y = y
        self.type = type
        self.map = map

    def __copy__(self):
        return type(self)(self.x, self.y, self.type, self.map)

    def can_move_up(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.VERTICAL_MINAUTORE:
            if not self.map.case_have_top_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x, self.y - 1):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.y - 1 == ariane.y and self.x == ariane.x \
                    and not self.map.case_have_top_wall(self.x, self.y):
                return True

        if self.type == Person.HORIZONTAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.x == ariane.x \
                    and not self.map.case_have_top_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x, self.y - 1):
                return True

        return False

    def move_up(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE or self.type == Person.THESEE:
            if self.y > 0:
                self.y -= 1

        if self.type == Person.HORIZONTAL_MINAUTORE or self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            thesee = self.map.get_thesee()

            while self.y > ariane.y:
                if self.y == thesee.y and self.x == thesee.x:
                    break

                if self.map.case_have_top_wall(self.x, self.y):
                    break

                if self.map.mino_on_case(self.x, self.y - 1):
                    break

                self.y -= 1

        return self

    def can_move_down(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.VERTICAL_MINAUTORE:
            if not self.map.case_have_bottom_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x, self.y + 1):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.y + 1 == ariane.y and self.x == ariane.x \
                    and not self.map.case_have_bottom_wall(self.x, self.y):
                return True

        if self.type == Person.HORIZONTAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.x == ariane.x \
                    and not self.map.case_have_bottom_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x, self.y + 1):
                return True

        return False

    def move_down(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE or self.type == Person.THESEE:
            if self.y < self.map.size - 1:
                self.y += 1

        if self.type == Person.HORIZONTAL_MINAUTORE or self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            thesee = self.map.get_thesee()

            while self.y < ariane.y:
                if self.y == thesee.y and self.x == thesee.x:
                    break

                if self.map.case_have_bottom_wall(self.x, self.y):
                    break

                if self.map.mino_on_case(self.x, self.y + 1):
                    break

                self.y += 1

        return self

    def can_move_left(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.HORIZONTAL_MINAUTORE:
            if not self.map.case_have_left_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x - 1, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.x - 1 == ariane.x and self.y == ariane.y \
                    and not self.map.case_have_left_wall(self.x, self.y):
                return True

        if self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.y == ariane.y \
                    and not self.map.case_have_left_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x - 1, self.y):
                return True

        return False

    def move_left(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE or self.type == Person.THESEE:
            if self.x > 0:
                self.x -= 1

        if self.type == Person.HORIZONTAL_MINAUTORE or self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            thesee = self.map.get_thesee()

            while self.x > ariane.x:
                if self.x == thesee.x and self.y == thesee.y:
                    break

                if self.map.case_have_left_wall(self.x, self.y):
                    break

                if self.map.mino_on_case(self.x - 1, self.y):
                    break

                self.x -= 1

        return self

    def can_move_right(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.HORIZONTAL_MINAUTORE:
            if not self.map.case_have_right_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x + 1, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.x + 1 == ariane.x and self.y == ariane.y \
                    and not self.map.case_have_right_wall(self.x, self.y):
                return True

        if self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.y == ariane.y \
                    and not self.map.case_have_right_wall(self.x, self.y) \
                    and not self.map.mino_on_case(self.x + 1, self.y):
                return True

        return False

    def move_right(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE or self.type == Person.THESEE:
            if self.x < self.map.size - 1:
                self.x += 1

        if self.type == Person.HORIZONTAL_MINAUTORE or self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            thesee = self.map.get_thesee()

            while self.x < ariane.x:
                if self.x == thesee.x and self.y == thesee.y:
                    break

                if self.map.case_have_right_wall(self.x, self.y):
                    break

                if self.map.mino_on_case(self.x + 1, self.y):
                    break

                self.x += 1

        return self

    def can_move_to(self, direction) -> bool:
        if direction == "Up":
            return self.can_move_up()
        elif direction == "Down":
            return self.can_move_down()
        elif direction == "Left":
            return self.can_move_left()
        elif direction == "Right":
            return self.can_move_right()

        return False

    def move_to(self, direction):
        if direction == "Up":
            self.move_up()
        elif direction == "Down":
            self.move_down()
        elif direction == "Left":
            self.move_left()
        elif direction == "Right":
            self.move_right()

        return self

    def get_type(self):
        return self.type
