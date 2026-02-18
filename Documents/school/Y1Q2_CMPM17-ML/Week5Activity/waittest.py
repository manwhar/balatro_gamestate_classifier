class Beans:
    def __init__(self, uwu, yay):
        self.uwu = uwu
        self.yay = yay

    def get_uwu(self):
        return self.uwu


bong = Beans("a", "b")
bean = Beans.__init__(self=bong, uwu="whee", yay="wahoo")
print(bong.get_uwu())
