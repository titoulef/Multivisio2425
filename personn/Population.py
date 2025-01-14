import personn

class Population():
    def __init__(self):
        self.population = []

    def addPerson(self, pers):
        self.population.append(pers)

    def __str__(self):
        population_str = "\n".join(str(pers) for pers in self.population)
        return f"Population:\n{population_str}" if self.population else "Population is empty."





