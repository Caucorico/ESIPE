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

    def can_move_up(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.VERTICAL_MINAUTORE:
            if not self.map.case_have_top_wall(self.x, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.y - 1 == ariane.y and self.x == ariane.x:
                return True

        if self.type == Person.HORIZONTAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.x == ariane.x \
                    and not self.map.case_have_top_wall(self.x, self.y):
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

                self.y -= 1

        return self

    def can_move_down(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.VERTICAL_MINAUTORE:
            if not self.map.case_have_bottom_wall(self.x, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.y + 1 == ariane.y and self.x == ariane.x:
                return True

        if self.type == Person.HORIZONTAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.x == ariane.x \
                    and not self.map.case_have_bottom_wall(self.x, self.y):
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

                self.y += 1

        return self

    def can_move_left(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.HORIZONTAL_MINAUTORE:
            if not self.map.case_have_left_wall(self.x, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.x - 1 == ariane.x and self.y == ariane.y:
                return True

        if self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.y == ariane.y \
                    and not self.map.case_have_left_wall(self.x, self.y):
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

                self.x -= 1

        return self

    def can_move_right(self):
        # Todo : find a way to polymorphism
        if self.type == Person.ARIANE \
                or self.type == Person.HORIZONTAL_MINAUTORE:
            if not self.map.case_have_right_wall(self.x, self.y):
                return True

        if self.type == Person.THESEE:
            ariane = self.map.get_ariane()
            if self.x + 1 == ariane.x and self.y == ariane.y:
                return True

        if self.type == Person.VERTICAL_MINAUTORE:
            ariane = self.map.get_ariane()
            if self.y == ariane.y \
                    and not self.map.case_have_right_wall(self.x, self.y):
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

                self.x += 1

        return self

    def get_type(self):
        return self.type
