class Case:

    def __init__(self, i, j):
        self.x = i
        self.y = j
        self.left_wall = False
        self.right_wall = False
        self.top_wall = False
        self.bottom_wall = False

    def set_left_wall(self, value):
        self.left_wall = value

    def set_right_wall(self, value):
        self.right_wall = value

    def set_top_wall(self, value):
        self.top_wall = value

    def set_bottom_wall(self, value):
        self.bottom_wall = value

    def has_left_wall(self):
        return self.left_wall

    def has_right_wall(self):
        return self.right_wall

    def has_top_wall(self):
        return self.top_wall

    def has_bottom_wall(self):
        return self.bottom_wall
