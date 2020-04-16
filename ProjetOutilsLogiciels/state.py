from __future__ import annotations

from person import Person


# Min python = 3.7

class State:

    def __init__(self, ariane: Person = None, thesee: Person = None, door: Person = None, minos_v: list = None,
                 minos_h: list = None):
        self.ariane = ariane
        self.thesee = thesee
        self.door = door

        if minos_v is None:
            self.minos_v = []
        else:
            self.minos_v = minos_v

        if minos_h is None:
            self.minos_h = []
        else:
            self.minos_h = minos_h

    def __eq__(self, other):
        if self.ariane.x != other.ariane.x:
            return False
        if self.ariane.y != other.ariane.y:
            return False
        if len(self.minos_h) != len(other.minos_h):
            return False
        for i in range(0, len(self.minos_h)):
            if self.minos_h[i].x != other.minos_h[i].x:
                return False
            if self.minos_h[i].y != other.minos_h[i].y:
                return False
        if len(self.minos_v) != len(other.minos_v):
            return False
        for i in range(0, len(self.minos_v)):
            if self.minos_v[i].x != other.minos_v[i].x:
                return False
            if self.minos_v[i].y != other.minos_v[i].y:
                return False
        if self.thesee.x != other.thesee.x:
            return False
        return self.thesee.y == other.thesee.y

    def __hash__(self):
        data_list = [self.ariane.x, self.ariane.y, self.thesee.x, self.thesee.y, self.door.x, self.door.y]
        compute_hash = 0

        for mino_h in self.minos_h:
            data_list.append(mino_h.x)
            data_list.append(mino_h.y)

        for mino_v in self.minos_v:
            data_list.append(mino_v.x)
            data_list.append(mino_v.y)

        for i in range(0, len(data_list)):
            compute_hash += data_list[i]*(10**i)

        return compute_hash

    def set_ariane(self, ariane: Person) -> State:
        self.ariane = ariane
        return self

    def get_ariane(self) -> Person:
        return self.ariane

    def set_thesee(self, thesee: Person) -> State:
        self.thesee = thesee
        return self

    def get_thesee(self) -> Person:
        return self.thesee

    def set_door(self, door: Person) -> State:
        self.door = door
        return self

    def get_door(self) -> Person:
        return self.door

    def set_minos_v(self, minos_v: list) -> State:
        self.minos_v = minos_v
        return self

    def get_minos_v(self) -> list:
        return self.minos_v

    def set_minos_h(self, minos_h: list) -> State:
        self.minos_h = minos_h
        return self

    def get_minos_h(self) -> list:
        return self.minos_h

    def add_mino_v(self, mino_v: Person) -> list:
        self.minos_v.append(mino_v)
        return self.minos_v

    def add_mino_h(self, mino_h: Person) -> list:
        self.minos_h.append(mino_h)
        return self.minos_h
