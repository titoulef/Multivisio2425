import personn

class Population():
    def __init__(self):
        self.pers = []
        self.suitcases = []
        self.lien_dict =  {}

    def addPerson(self, pers):
        self.pers.append(pers)

    def addSuit(self, suit):
        self.suitcases.append(suit)

    def addLien(self, ID_suit, ID_pers):
        self.lien_dict[ID_pers] = ID_suit

    def __str__(self):
        population_str = "\n".join(str(pers) for pers in self.pers)
        suitcases_str = "\n".join(str(suit) for suit in self.suitcases)
        liens_str = "\n".join(
            f"Person ID: {pers_id} -> Suitcase ID: {suit_id}" for pers_id, suit_id in self.lien_dict.items())

        result = "Population: "
        result += population_str if self.pers else "No persons."
        result += "\nSuitcases: "
        result += suitcases_str if self.suitcases else "No suitcases."
        result += "\nLinks: "
        result += liens_str if self.lien_dict else "No links."

        return result



