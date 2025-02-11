import numpy as np
from utils.bbox_utils import bbox_distance

# Transition matrix (probabilités d'association)
#   État 0 : Personne n'est pas associée à la valise
#   État 1 : Personne est associée à la valise
transition_matrix = np.array([
    [0.9, 0.1],  # Probabilités de transition de l'état 0 -> 0 (0.9), de l'état 0 -> 1 (0.1)
    [0.3, 0.7]   # Probabilités de transition de l'état 1 -> 0 (0.3), de l'état 1 -> 1 (0.7)
])

# Probabilités initiales (basées sur les premières observations)
initial_probabilities = np.array([0.5, 0.5])  # Exemple : 50% chance que la valise soit associée

def update_association(probabilities, observation):
    """
    Met à jour les probabilités d'association avec la valise
    :param probabilities: probabilité actuelle d'être dans chaque état (non associé, associé)
    :param observation: mesure (par exemple, distance ou couverture de la bbox)
    :return: nouvelle probabilité d'association
    """
    # Mise à jour de la probabilité en fonction de l'observation
    # Exemple simple : si l'observation est faible, on suppose une probabilité plus forte d'association
    if observation < 0.5:  # Si la distance entre la personne et la valise est faible
        observation_prob = 0.9  # Forte probabilité d'association
    else:
        observation_prob = 0.1  # Faible probabilité d'association

    # Appliquer la matrice de transition pour mettre à jour l'état
    new_probabilities = np.dot(probabilities, transition_matrix)

    # Appliquer l'observation à la probabilité d'association
    new_probabilities[1] *= observation_prob  # Mettre à jour la probabilité d'association
    new_probabilities[0] *= (1 - observation_prob)  # Mettre à jour la probabilité de non-association

    # Normaliser les probabilités pour qu'elles somme à 1
    new_probabilities /= new_probabilities.sum()

    return new_probabilities

def predict_association(player_id, suitcase_id, player_data, suitcase_data):
    # Récupérer les données de la personne et de la valise
    data_person = player_data[player_id]
    bbox_person = data_person['bbox']
    bbox_suitcase = suitcase_data[suitcase_id]

    # Calculer la distance entre la personne et la valise (par exemple, avec bbox_distance)
    distance = bbox_distance(bbox_person, bbox_suitcase)

    # Mettre à jour la probabilité d'association avec la valise
    prob = initial_probabilities  # Probabilité initiale
    updated_prob = update_association(prob, distance)

    # Si la probabilité d'association est plus grande que 0.5, la valise appartient à la personne
    if updated_prob[1] > 0.5:
        return True  # La valise appartient à la personne
    else:
        return False  # La valise n'appartient pas à la personne

# Exemple d'utilisation
player_data = {
    1: {'bbox': [100, 200, 150, 250]},  # La boîte englobante de la personne
}
suitcase_data = {
    1: {'bbox': [120, 220, 170, 270]},  # La boîte englobante de la valise
}

# Essayer de prédire si la valise 1 appartient à la personne 1
result = predict_association(1, 1, player_data, suitcase_data)
print("La valise appartient à la personne:", result)
