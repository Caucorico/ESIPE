from __future__ import annotations

from person import Person
from copy import deepcopy
from copy import copy


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
