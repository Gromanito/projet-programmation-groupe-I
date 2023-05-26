from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('/home/romain/Perso/Cours/L3_deuxieme_semestre/projet-Programmation/TestYolo/runs/classify/train14/weights/best.pt')


results = model.predict(source='/home/romain/Perso/Cours/L3_deuxieme_semestre/projet-Programmation/src/pourYOLO/imagePredite') # source already setup

# print(results.probs)
# print(results)
# print(dir(results[0]))
# print(results[0].probs.numpy())
listeReconnus = []
for r in results:
	array = r.probs.numpy()
	maxConfiance=0
	indice=-1
	for i, proba in enumerate(array):
		if proba > maxConfiance:
			indice, maxConfiance = i, proba
	listeReconnus.append(model.names[indice])

print(listeReconnus)
