"""  
fonctions utiles pour transformer une image en quelque chose d'utile pour YOLO

les fonctions prennent une image en entrée, elle doit être binarisée et en niveau de gris
(voir fonction  cv.cvtColor(img, cv.COLOR_BGR2GRAY)  

"""


from ultralytics import YOLO
#assert <img> is not None,  "Message qui dit que ça a pas marché"

import cv2 as cv
import os
import math # pour les radians et les cosinus
import numpy as np
import shutil #pour supprimer les dossiers récursivement (faire attention)

from segmente import *


SEUIL_BASIQUE = 0
SEUIL_ADAPTATIF = 1
SEUIL_ADAPTATIF_GAUSSIEN = 2




def binarisation(img, typeBinarisation = 0, Seuil=128, Otsu=False, dejaNDG=False, filtreGaussien=False):
	
	if not dejaNDG:
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	masqueFeuilleBlanche = enleveFondNoir(img)


	match typeBinarisation:
		case 0 :#SEUIL_BASIQUE:
			if filtreGaussien:
				img = cv.GaussianBlur(img, (5,5), 0)
			_, img = cv.threshold(img, Seuil, 255, cv.THRESH_BINARY)
		
		case 1 :#SEUIL_ADAPTATIF:
			img = cv.adaptiveThreshold(img, 255 , cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11 , 2) 
		
		case 2 :#SEUIL_ADAPTATIF_GAUSSIEN:
			img = cv.adaptiveThreshold(img, 255 , cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11 , 2)
		
		case 3 :#SEUIL_SAUVOLA:
			img = cv.bitwise_not(img) #faut inverser pour sauvola, c'comme ça
			img = cv.ximgproc.niBlackThreshold(img, 255, type=cv.ximgproc.BINARIZATION_SAUVOLA, blockSize=165, k=0.8)

			
		case 4 :#appliquer le filtre de base, bon jsp:
			img = appliquerBinarisationParDefaut(img)
			
		case _:
			print("mauvais argument pour typeBinarisation, aucun seuillage appliqué")
	
	if Otsu: # on garde que la feuille, on met le fond en blanc
		img = cv.bitwise_or(img, masqueFeuilleBlanche)
	return img
	


def enleveFondNoir(img):
	#image est déjà en NDG

	# on va appliquer un filtre de otsu, qui distingue 2 types d'objets (nous ce sera feuille blanche / fond noir)
	#probleme : si y a pas de fond noir, ça enlève l'écriture
	# on rajoute manuellement du fond noir avec une translation (d'où la matrice)
	blur = cv.GaussianBlur(img, (25,25), 0) # "égalise la feuille"

	M = np.float32([
	[1, 0, 300],  #translation vers la droite
	[0, 1, 300]		#translation vers le bas
	])
	agrandi = cv.warpAffine(blur, M, (img.shape[1]+600, img.shape[0]+600), borderValue=(127,127,127))
	

	_, feuilleBlanche = cv.threshold(agrandi, 127 ,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	masqueFeuilleBlanche = cv.bitwise_not(feuilleBlanche)	#on a que la feuille blanche normalement
	
	#y a moyen que les lettres aient étaient sélectionnées comme fond, donc on fait des fermetures 
	masqueFeuilleBlanche = ouverture(masqueFeuilleBlanche, 9)

	masqueFeuilleBlanche = masqueFeuilleBlanche[300: masqueFeuilleBlanche.shape[0]-300,  300: masqueFeuilleBlanche.shape[1]-300] # on redimensionne bien
	if masqueFeuilleBlanche.shape[0] != img.shape[0]:
		print("pas bon")

	

	return masqueFeuilleBlanche
	#retourne le masque où y a que la feuille de blanc


def ouverture(img, nombreErosion, tailleMatrice=5):
	for i in range(nombreErosion):
		img = erosion(img,tailleMatrice)
	for i in range(nombreErosion-3): # on fait -3 pour pas avoir trop de bordure
		img = dilatation(img, tailleMatrice) 
	return img




def dilatation(img, tailleMatrice=5):
    imgOut = img.copy()
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE,(tailleMatrice,tailleMatrice))
    imgOut = cv.dilate(imgOut, kernel_open)
    return imgOut



def erosion (img, tailleMatrice=5):
    imgOut = img.copy()
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE,(tailleMatrice,tailleMatrice))
    imgOut = cv.erode(imgOut, kernel_close)
    return imgOut



def compteNbErosion(img):
	
	img = cv.bitwise_not(img) #on inverse prck faut compter le nombre de pixels noir, pas blanc
	
	compteurNberosion = 0
	while cv.countNonZero(img) != 0: #ttque il reste du blanc
		img = erosion(img)
		compteurNberosion += 1
	
	return compteurNberosion



def dilateCommeIlFaut(img):
	""" 
	dilate (ou rétrécit) le bon nombre de fois pour que YOLO arrive mieux à deviner
	il faut que l'image entrée soit binarisée
	"""

	#on rajoute une bordure blanche autour prck des fois la lettre est collée au bord et ça demande + d'érosion

	M = np.float32([
	[1, 0, 1],  #translation vers la droite
	[0, 1, 1]		#translation vers le bas
	])
	img = cv.warpAffine(img, M, (img.shape[1]+2, img.shape[0]+2), borderValue=(255,255,255))
	

	nberosionAFaire = 7 - compteNbErosion(img)
	# 6 correspond en gros à l'épaisseur des images utilisée pour entrainer YOLO

	if nberosionAFaire < 0: # le trait est trop épais, on augmente le blanc ( et réduit le noir, i.e. la lettre)
		for i in range(-1 * nberosionAFaire):
			img=dilatation(img)  
	else:
		for i in range(nberosionAFaire): # le trait est pas assez épais, on réduit le blanc (et on dilate le noir, i.e. la lettre)
			img=erosion(img) 
	
	return img[1:img.shape[0]-1, 1:img.shape[1]-1]


def redimensionneImageEn128SansBord(img):
	""" rogne l'image en supprimant les bandes blanches du bord de l'image,
		et redimensionne l'image en 128x128, en respectant à peu près l'échelle
		(l'image est normalement binarisée)
		"""
	
	img = cv.bitwise_not(img) #on inverse prck pour opencv le fond est noir et l'objet est blanc
	x,y,w,h = cv.boundingRect(img) # on prend le rectangle de la zone d'intéret
	img = img[y:y+h, x:x+w]  # on crop
	imgAvantResize = cv.bitwise_not(img) #on reinverse pour avoir la lettre en noir (c'est + beau)

	img = cv.resize(imgAvantResize, (128,128), interpolation = cv.INTER_NEAREST_EXACT ) #on resize comme il faut en 128
	#INTER_NEAREST_EXACT est la meilleure interpolation qu'on ait testée (floute pas et respecte bien l'échelle)

	_, img = cv.threshold(img, 10, 255, cv.THRESH_BINARY) # on seuil prck ptre que ça floute de resize

	return img



def rendPretPourYolo(cheminVersDossier):
	for filename in os.listdir(cheminVersDossier):
		if filename.endswith(".png"):
			image = cv.imread(cheminVersDossier+'/'+filename, cv.IMREAD_GRAYSCALE)
			image = redimensionneImageEn128SansBord(image)
			image = dilateCommeIlFaut(image)
			cv.imwrite(cheminVersDossier+'/'+filename, image)


#fonction qui rotate une image
def rotation(image, angleInDegrees, boundingBox=True):
	h, w = image.shape[:2]
	img_c = (w / 2, h / 2)

	rot = cv.getRotationMatrix2D(img_c, angleInDegrees, 1)

	if boundingBox:

		rad = math.radians(angleInDegrees)
		sin = math.sin(rad)
		cos = math.cos(rad)
		w, h = int((h * abs(sin)) + (w * abs(cos))), int((h * abs(cos)) + (w * abs(sin)))

		rot[0, 2] += ((w / 2) - img_c[0])
		rot[1, 2] += ((h / 2) - img_c[1])
	

	outImg = cv.warpAffine(image, rot, (w, h), borderValue=(255,255,255), flags=cv.INTER_LINEAR)
	#l'image est pivotée, on transforme le fond noir en fond blanc


	return outImg


# pris de la page https://stackoverflow.com/questions/23195522/opencv-fastest-method-to-check-if-two-images-are-100-same-or-not
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
	




#tiré du site : https://stackoverflow.com/questions/73232684/how-to-segment-text-handwritten-lines-using-horizontal-profile-projection
# Function to generate horizontal projection profile
def getHorizontalProjectionProfile(image):
	#l'image doit déjà être binarisée
    # convert black spots to ones and others to zeros
    binary = np.where(image == 0, 1, 0)
    # add up rows
    horizontal_projection = np.sum(binary, axis=1)
    return horizontal_projection


def trouveMeilleureAngle(image):

	"""	
	image est déjà binarisée
	trouve le meilleur angle de rotation entre -30 et 30 degrés (faut pas abuser)
	marche bien pour seuil adaptatif gaussien après avoir fait otsu
	
	"""
	maxHisto=0
	maxAngle = -30

	for i in range(-25, 25):
		imageRotate = binarisation(rotation(image, i, False), Seuil=127, dejaNDG=True )
		
		histo = getHorizontalProjectionProfile(imageRotate)
		
		
		maxHorizontalProjectProfile = histo.max()
		if maxHisto< maxHorizontalProjectProfile:
			maxHisto = maxHorizontalProjectProfile
			maxAngle = i
	
	return maxAngle



def appliquerBinarisationParDefaut(image):
	#l'image entrée est la photo prise par l'appareil

	#blabla ça prend une image et ça l'arrange le mieux possible
	""" truc cool à faire si on a le temps:
		détecter seulement la feuille (avec otsu)
		rogner comme il faut (doit pas être compliqué après binarisation)

		à la fin ça retourne une image binaire '''''''''prete''''''' pour la segmentation
	
	"""
	if len(image.shape) == 3:
		image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

	masqueFeuilleBlanche = enleveFondNoir(image) #on garde le masque pour la fin



	imageSeuilAdapt = image.copy()

	for i in range(5):#en général en appliquant 5 flou median et 1 seuillage adaptatif c'était stylé
		imageSeuilAdapt =  cv.medianBlur(imageSeuilAdapt, 3)



	#première binarisation pour trouver le meilleur angle pour la segmentation (on enlève le fond pour pas être embêté)
	imageSeuilAdapt = binarisation(imageSeuilAdapt, typeBinarisation=1, dejaNDG=True) 
	# cv.imwrite("seuillageAda.png", imageSeuilAdapt)

	#les paramètres que j'ai trouvé à peu près bien
	imageSauvola = cv.bitwise_not(image) #faut inverser pour sauvola, c'comme ça
	imageSauvola = cv.ximgproc.niBlackThreshold(imageSauvola, 255, type=cv.ximgproc.BINARIZATION_SAUVOLA, blockSize=165, k=0.8)
	# cv.imwrite("seuillageSau.png", imageSauvola)

	image = cv.bitwise_or(imageSauvola, imageSeuilAdapt)
	# cv.imwrite("avantMasque.png", image)
	image = cv.bitwise_or(image, masqueFeuilleBlanche)
	# cv.imwrite("ApresMasque.png", image)


	angleOptimal = trouveMeilleureAngle(image)


	image = rotation(image, angleOptimal, True)
	#la rotation transforme certains pixels en pas binaires, on rebinarise avec un seuil "qui marche"
	image = binarisation(image, Seuil=127, dejaNDG=True)

	#on a l'image à peu près correcte mais possiblement les bordures de la feuille
	#faudrait crop....... (ou pas)

	return angleOptimal, image





def enlevePointsNoirs(image):
	#image est binarisée7

	imageOuverte = dilatation(image, 3) # enleve un peu de point noir

	imageMasque = erosion(erosion(imageOuverte, 3), 3)

	imgSansPoint = cv.bitwise_or(image ,imageMasque)

	return imgSansPoint











def segmentationLigneRomain(image):
	#l'image est binarisée et en niveau de gris mais y a du bruit poivre et sel

	# image = dilatation(image, 3)

	"""  implémentation bizarre:
		on parcourt les 5 premières lignes, et on additionne le nombre de leur pixel
		ensuite on prend la ligne d'après et on fait la même chose avec les 4 de ligne d'avant
		abcde -> bcdef -> cdefg etc
		pour chaque 5uplets de lignes, on somme le nombre de pixel et on regarde si c'est 
		au dessus de la moyenne
		si oui, il s'agit d'une ligne, sinon non

	  """
	 # Invert


	image = 255 - image

	# Calculate horizontal projection
	proj = np.sum(image,1)

	ligneMax = np.amax(proj)
	lignePotentielle = []

	

	for row in range(0, image.shape[0]):
		
		if proj[row] > ligneMax * 0.1:
			lignePotentielle.append(row)

	#on sait qu'on a environ que les lignes, on calcule la moyenne des lignes
	moyenne = sum(proj[i] for i in lignePotentielle) / len(lignePotentielle)



	listeDesLignes = []
	enTrainDeLireUneLigne = False

	#on considère que les lignes en dessous de ligneMax sont des espaces, on regarde que ces lignes pour l'instant

	for row in range (0, image.shape[0]-9 ):
		moyenneDesCinqLignes = sum([proj[row+i]  for i in range(10)]) / 10
		if moyenneDesCinqLignes > moyenne * 0.15: # on vient de trouver une candidate
			if not enTrainDeLireUneLigne:
				enTrainDeLireUneLigne = True
				listeDesLignes.append([row])
		else:
			if enTrainDeLireUneLigne:
				enTrainDeLireUneLigne = False
				listeDesLignes[-1].append(row+4)


	
	#la dernière row analysée correspondait bien à une ligne, on la rajoute (prck la boucle du haut ne le fait pas)
	if enTrainDeLireUneLigne:
		listeDesLignes[-1].append(image.shape[0]-1)



	listeDesLignesOk = [listeDesLignes[0]]
	del listeDesLignes[0]
	#si des lignes sont "collées" on les réassemble (genre une ligne [5-15] et une autre [16 - 42] en fait c'est la meme ligne)
	for liste in listeDesLignes:
		if liste[0] - listeDesLignesOk[-1][1] < 4: #les lignes sont proches
			listeDesLignesOk[-1][1] = liste[1]
		else:
			listeDesLignesOk.append(liste) # les lignes sont pas proches
	
	
	#on supprime les lignes qui font moins de 7 pixels (surement du bruit et de tt façon 7 pixels pas analysable par yolo)
	listeDesLignes = []
	for ligne in listeDesLignesOk:
		if ligne[1] - ligne[0] > 7 :
			listeDesLignes.append(ligne)

	# #on augmente un peu la hauteur des lignes prck mon implémentation évite facilement les bordures des lignes
	# for i, ligne in enumerate(listeDesLignes):
	# 	hauteurLigne= ligne[1]-ligne[0]
	# 	listeDesLignes[i] = [ round(ligne[0]-hauteurLigne*0.10), round(ligne[1]+hauteurLigne*0.10)]
	

	return listeDesLignes










def segmentationLettreRomain(image):

	#marche bien si les lettres ont à peu près la meme taille 

	#faudrait calculer la meilleure rotation prck ptre que l'écriture est penchée
	#meilleur angle blabla


	"""
	implémentation bizarre aussi

	on compte la moyenne du nombre de pixel par colonne

	on fait la moyenne de chaque 3 colonnes voisines. si c'est au dessus du nombre moyen, c'est une lettre, sinon un espace
	
	possible qu'il y ait 2 lettres très proches qui sont comptées comme une seule.
	on regarde pour chaque 2 lettres est ce que les 2 lettres + l'espace est inférieur bcp
	aux autres lettres, si oui -> C'est que c'est une seule lettre


	"""



	#l'image est binarisée, il s'agit de l'image d'une ligne
	 # Invert
	image = 255 - image

	# Calculate vertical projection





	proj = np.sum(image,0)


	total = np.sum(proj)
	moyenne = (total/proj.shape[0] ) * 0.1

	#on fait un premier passage où on récupère QUE les lignes qui sont potentiellement des lettres (au dessus de la moyenne)
	#on recompte le nombre de ces potentielles lettres pour faire une moyenne sur le nombre de pixel de ces lettres



	listeDesLettres = []
	enTrainDeLireUneLettre = False


	for colonne in range (1, image.shape[1]-1 ):
		moyenneDesTroisColonnes = sum([proj[colonne+i]  for i in range(-1,2)]) / 3
		if moyenneDesTroisColonnes > moyenne: # on vient de trouver une candidate
			if not enTrainDeLireUneLettre:
				enTrainDeLireUneLettre = True
				listeDesLettres.append([colonne])
		else:
			if enTrainDeLireUneLettre:
				enTrainDeLireUneLettre = False
				listeDesLettres[-1].append(colonne)
	
	#la dernière colonne analysée correspondait bien à une lettre, on la rajoute (prck la boucle du haut ne le fait pas)
	if enTrainDeLireUneLettre:
		listeDesLettres[-1].append(image.shape[1]-1)

	

	enTrainDeLireUneLettre = False

	moyenneLettre = sum([ sum(proj[lettre[0]:lettre[1]])  for lettre in listeDesLettres]) /  sum(lettre[1]-lettre[0] for lettre in listeDesLettres)
	# print(moyenneLettre)
	listeDesLettres.clear()
	#deuxieme passage où on essaie de voir les lettres avec la "vraie moyenne" des lettres
	#en gros pour rendre "équivalent" les textes qui ont des gros espaces et ceux qui en ont des petits
	for colonne in range (2, image.shape[1]-2 ):
		moyenneDesTroisColonnes = sum([proj[colonne+i]  for i in range(-2,3)]) / 5
		if moyenneDesTroisColonnes > moyenneLettre * 0.02 : # on vient de trouver une candidate
			if not enTrainDeLireUneLettre:
				enTrainDeLireUneLettre = True
				listeDesLettres.append([colonne])
		else:
			if enTrainDeLireUneLettre:
				enTrainDeLireUneLettre = False
				listeDesLettres[-1].append(colonne)
	
	#la dernière colonne analysée correspondait bien à une lettre, on la rajoute (prck la boucle du haut ne le fait pas)
	if enTrainDeLireUneLettre:
		listeDesLettres[-1].append(image.shape[1]-1)



	#on sépare les lettres trop longues en 2 lettres
	"""
	longueurTotaleDesLettres = sum([liste[1]-liste[0] for liste in listeDesLettres] )
	moyenneLongueurLettre = longueurTotaleDesLettres / len(listeDesLettres)

	listeDesLettresOk=[]
	
	for lettre in listeDesLettres:
		longueur2lettres = lettre[1] - lettre[0]

		if longueur2lettres > 1.35 * moyenneLongueurLettre:#on casse la lettre en deux en prenant le plus petit point
			#on trouve l'endroit de la plus petite colonne
			minimumVal = proj[lettre[0]+3]
			minColonne = lettre[0]
			for numCol in range(lettre[0]+3, lettre[1]-3):
				if minimumVal > proj[numCol] :
					minimumVal = proj[numCol]
					minColonne = numCol
					
			listeDesLettresOk.append([lettre[0], minColonne])
			listeDesLettresOk.append([minColonne, lettre[1]])
		else:
			listeDesLettresOk.append(lettre)
	"""
	# y a des lettres mal écrites qui prennent 2 lettres mais sont très proches : c'estl la même lettre
	# idée pour les reconnaitre: si c'est la meme lettre l'espacement qu'il y a entre les 2 composantes
	# est très faible comparé à l'espacement à gauche et à droite, on regarde pour chaque couple de lettre 
	# si l'espacement est grand ou pas comparé 


	



	
	return listeDesLettres







def getVerticalProjectionProfile(image):
	return (getHorizontalProjectionProfile(image.transpose()))
	


# def rogneBordures(image):
# 	return
# 	#l'image est déjà binarisée et en niveau de gris




def ecritImagetteDansFichier(imagettes): # liste de liste d'image
	chemin="images/imageSegmentee"

	try: # on remplace les anciennes images segmentées par les nouvelles
		os.mkdir(chemin) # le dossier n'existe pas
	except OSError as error: 
		shutil.rmtree(chemin)
		os.mkdir(chemin) # le dossier n'existe pas

	for i, ligne in enumerate(imagettes):

		nomDossierLigne = chemin + "/ligne{:02d}".format(i)
		os.mkdir(nomDossierLigne)

		for j, lettre in enumerate(ligne):

			
			cv.imwrite(nomDossierLigne + "/lettre{:02d}.png".format(j), lettre)




# for filename in os.listdir("images/imagesTestYOLO/img"):
# 	print(filename)
# 	_, _, _, imagettes = segmentation(cv.imread("images/imagesTestYOLO/img/" + filename))
# 	ecritImagetteDansFichier(imagettes)


# for filename in os.listdir("images/imagesTestYOLO/img"):
# 	print(filename)
# 	imagettes = segmenteRomain(cv.imread("images/imagesTestYOLO/img/"+filename, cv.IMREAD_GRAYSCALE))
# 	ecritImagetteDansFichier(imagettes)




"""petit programme qui "compte" la fiabilité de yolo"""

def compteCombienCoherent():
	model = YOLO("src/pourYOLO/entrainementsYolo/bestUpperSansBordDilate.pt")
	for lettre in os.listdir("images/imagesTestYOLO/imagesDeLaurent"):
		
		rendPretPourYolo("images/imagesTestYOLO/imagesDeLaurent/"+lettre)

		compteurJuste = 0
		texteAEcrire = []
		results = model.predict("images/imagesTestYOLO/imagesDeLaurent/"+ lettre, verbose = False)
		for r in results:
			probas = r.probs.numpy().tolist() # on chope la liste des probas
			dico = [(model.names[i], proba ) for i, proba in enumerate(probas) ]
			cinqMeilleuresProbas = sorted(dico, key=lambda x:x[1], reverse=True)[:5]
			if cinqMeilleuresProbas[0][0] == lettre:
				compteurJuste += 1
			texteAEcrire.append(r.path[-12:])
			for i in range(5):
				texteAEcrire.append(cinqMeilleuresProbas[i][0] + '   ' + str(cinqMeilleuresProbas[i][1]) )
			texteAEcrire.append('\n')
		with open("images/imagesTestYOLO/imagesDeLaurent/" +lettre + '/' +  lettre + ".txt", "w") as file:
			file.write( str(compteurJuste)+"/12 corrects\n\n" + '\n'.join(texteAEcrire))

		


# model = YOLO("src/pourYOLO/entrainementsYolo/bestUpperSansBord.pt")

# model.predict("images/imageSegmentee/ligne00")

