import upemtk
import sys
from person import *
from PIL import Image
from shutil import copyfile


class Draw:

    def __init__(self, size, map_size):
        self.size = size
        self.map_size = map_size
        self.resolution = (int((self.size - 20) / self.map_size), int((self.size - 20) / self.map_size))

    def get_images(self):
        self.ARIANE_IMAGE_PATH = "./tmp/ariane.png"
        self.THESEE_IMAGE_PATH = "./tmp/thesee.png"
        self.DOOR_IMAGE_PATH = "./tmp/porte.png"
        self.MINO_H_IMAGE_PATH = "./tmp/minoH.png"
        self.MINO_V_IMAGE_PATH = "./tmp/minoV.png"

        copyfile("./media/ariane.png", self.ARIANE_IMAGE_PATH)
        copyfile("./media/thesee.png", self.THESEE_IMAGE_PATH)
        copyfile("./media/porte.png", self.DOOR_IMAGE_PATH)
        copyfile("./media/minoH.png", self.MINO_H_IMAGE_PATH)
        copyfile("./media/minoV.png", self.MINO_V_IMAGE_PATH)

        try:
            ariane = Image.open("./tmp/ariane.png", "r")
            thesee = Image.open("./tmp/thesee.png", "r")
            door = Image.open("./tmp/porte.png", "r")
            mino_h = Image.open("./tmp/minoH.png", "r")
            mino_v = Image.open("./tmp/minoV.png", "r")
        except FileNotFoundError:
            print("Fail")
            exit(1)

        ariane = ariane.resize(self.resolution)
        thesee = thesee.resize(self.resolution)
        door = door.resize(self.resolution)
        mino_h = mino_h.resize(self.resolution)
        mino_v = mino_v.resize(self.resolution)

        ariane.save(self.ARIANE_IMAGE_PATH)
        thesee.save(self.THESEE_IMAGE_PATH)
        door.save(self.DOOR_IMAGE_PATH)
        mino_h.save(self.MINO_H_IMAGE_PATH)
        mino_v.save(self.MINO_V_IMAGE_PATH)

    def start(self):
        upemtk.cree_fenetre(self.size, self.size)
        self.get_images()  # TODO : FIND A BETTER WAY TO INIT THIS IMAGES

    def stop(self):
        upemtk.ferme_fenetre()

    def draw_extern_wall(self):
        upemtk.ligne(10, 10, self.size - 10, 10, couleur='black', epaisseur=2)
        upemtk.ligne(10, 10, 10, self.size - 10, couleur='black', epaisseur=2)
        upemtk.ligne(self.size - 10, 10, self.size - 10, self.size - 10, couleur='black', epaisseur=2)
        upemtk.ligne(10, self.size - 10, self.size - 10, self.size - 10, couleur='black', epaisseur=2)

    def get_dimensions(self, j, i):
        left = 10 + i * ((self.size - 20) / self.map_size)
        right = 10 + (i + 1) * ((self.size - 20) / self.map_size)
        top = 10 + j * ((self.size - 20) / self.map_size)
        bottom = 10 + (j + 1) * ((self.size - 20) / self.map_size)

        return top, right, bottom, left

    def draw_wall(self, map):
        for i in range(0, map.get_size()):
            for j in range(0, map.get_size()):
                (top, right, bottom, left) = self.get_dimensions(j, i)

                if map.case_have_top_wall(i, j):
                    upemtk.ligne(left, top, right, top, couleur='black', epaisseur=2)

                if map.case_have_bottom_wall(i, j):
                    upemtk.ligne(left, bottom, right, bottom, couleur='black', epaisseur=2)

                if map.case_have_left_wall(i, j):
                    upemtk.ligne(left, top, left, bottom, couleur='black', epaisseur=2)

                if map.case_have_right_wall(i, j):
                    upemtk.ligne(right, top, right, bottom, couleur='black', epaisseur=2)

    def draw_entity(self, entity: Person):
        (top, right, bottom, left) = self.get_dimensions(entity.x, entity.y)

        if entity.get_type() == Person.ARIANE:
            upemtk.image(top, left, self.ARIANE_IMAGE_PATH, ancrage='nw')
        elif entity.get_type() == Person.THESEE:
            upemtk.image(top, left, self.THESEE_IMAGE_PATH, ancrage='nw')
        elif entity.get_type() == Person.PORTE:
            upemtk.image(top, left, self.DOOR_IMAGE_PATH, ancrage='nw')
        elif entity.get_type() == Person.VERTICAL_MINAUTORE:
            upemtk.image(top, left, self.MINO_V_IMAGE_PATH, ancrage='nw')
        elif entity.get_type() == Person.HORIZONTAL_MINAUTORE:
            upemtk.image(top, left, self.MINO_H_IMAGE_PATH, ancrage='nw')

    def draw_laby(self, labyrinth, entities):
        upemtk.rectangle(0, 0, self.size, self.size, couleur='', remplissage='white')
        self.draw_extern_wall()
        self.draw_wall(labyrinth)

        for entity in entities:
            self.draw_entity(entity)
