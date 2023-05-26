"""
segmente l'image entrée en paramètre et crée un dossier imageSegmentee
qui contient chaque caractère avec comme nom :

<ligne>_<colonne>_lettre.png

où ligne est le numéro de la ligne à laquelle se situe la lettre 
et colonne qui est la position de la lettre dans cette ligne
"""



import os #pour créer dossier
import sys #pour sys.argv
import cv2
import numpy as np


def coordonneeLigne(myprojection):
    danslaligne = False

    # on calcule les coordonnees de la delimitation des lignes
    coordDF = []
    for i, value in enumerate(myprojection):
        if value!=0:
            if not danslaligne or (danslaligne and i == len(myprojection)-1): 
                coordDF.append(i)
                if i == len(myprojection)-1 and len(coordDF)%2!=0:
                    coordDF.pop(-1)
                
            danslaligne = True
        else : 
            if danslaligne : coordDF.append(i)
            danslaligne=False
    
    lignes = [(coordDF[i],coordDF[i+1]) for i in range(0, len(coordDF),+2)]
    
    # on cheque si la taille des lignes sont cohérantes
    moyHauteurLignes = np.sum([c[1]-c[0] for c in lignes])/len(lignes)
    for j in range(len(lignes)-1,-1,-1):
        if lignes[j][1]-lignes[j][0]<moyHauteurLignes*0.1:
                lignes.pop(j)

    return lignes

def segmentLigne(image,width,coord_ligne):
    lignes = []
    
    for coordonnee in coord_ligne:
        uneligne = image[coordonnee[0]:coordonnee[1], 0:width]
        lignes.append(uneligne)
    return lignes

def coordonneeCaractere(myprojection):

    # on calcule les coordonnees de la delimitation des caracteres
    danslecaractere = False
    coordDF = []
    for i, value in enumerate(myprojection):
        if value!=0:
            if not danslecaractere or (danslecaractere and i == len(myprojection)-1): 
                coordDF.append(i)
                if i == len(myprojection)-1 and len(coordDF)%2!=0:
                    coordDF.pop(-1)
            danslecaractere = True
        else : 
            if danslecaractere : coordDF.append(i)
            danslecaractere=False
    
    lettres = [(coordDF[i],coordDF[i+1]) for i in range(0, len(coordDF),+2)]

    # on cheque si les espaces entre les lettres sont cohérantes
    if len(lettres)>1:
        espaceEntreLettre = [lettres[i+1][0]-lettres[i][1] for i in range(len(lettres)-1)]
        listePoidsEspace = espaceEntreLettre.copy()
        listePoidsEspace.sort() 
        listePoidsEspace = [x * (len(listePoidsEspace)-i) for i, x in enumerate(listePoidsEspace)]
        moyEspace = sum(listePoidsEspace)/(len(listePoidsEspace)/2 * (1+len(listePoidsEspace)))
        j = 0
        while (espaceEntreLettre != []):
            if espaceEntreLettre[0] < moyEspace*0.15:
                D = lettres.pop(j)
                F = lettres.pop(j)
                lettres.insert(j,(D[0],F[1]))
                espaceEntreLettre.pop(0)
            else :
                j += 1
                espaceEntreLettre.pop(0)
    return lettres

def trillageCaractere(img_lettres):

    # on calcule la hauteur des lettres grâce à la projection vertical 
    for i,img_l in enumerate(img_lettres):
        listeHauteurLettres = []
        moyenneHauteurLettres = 0
        nombreDeLettre = len(img_l)
        for img in img_l:
            hauteurLettre = 0
            _, threshold_image = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)

            # on calcule l'histogramme vertical de chaque lettre
            horizontal_pixel_sum = np.sum(threshold_image, axis=1)
            myprojection = horizontal_pixel_sum / 255

            # si une ligne n'est pas entièrement blanche on ajoute (MHL+=1)/nbrLettres
            for x in myprojection:
                if x != 0: 
                    hauteurLettre += 1
            moyenneHauteurLettres += hauteurLettre/nombreDeLettre
            listeHauteurLettres.append(hauteurLettre)
        
        # on évalue si la hauteur de la lettre est pertinante si non on supprime l'image de la lettre 
        for j in range(len(img_lettres[i])-1,-1,-1):
            if listeHauteurLettres[j]<moyenneHauteurLettres*0.2:
                img_lettres[i].pop(j)
    return img_lettres


# retourne coord_line [(deb1,fin1),(deb2,fin2)...], img_ligne sous forme [line1,line2,...], coord_caractere[[(deb11,fin11),(deb12,fin12)],[...],...], img_caract sous forme [[caracLigne1,...],[caracLigne2,...]]
def segmentation(image): 

    height, width, _ = image.shape

    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_scale, 125, 255, cv2.THRESH_BINARY_INV)

    # --------SEGMENTATION DES LIGNES--------
    # on calcule l'histogramme des lignes
    horizontal_pixel_sum = np.sum(threshold_image, axis=1)
    myprojectiony = horizontal_pixel_sum / 255
    coord_ligne = coordonneeLigne(myprojectiony)
    # on segmente l'image pour avoir des images contenant une ligne 
    img_ligne = segmentLigne(gray_scale,width,coord_ligne)

    # -----JUSTE DES LIGNES DE CODE POUR SAVE LES LIGNES-----
    for i, img in enumerate(img_ligne):
        nomImg = "line/line" + str(i) + ".png"
        cv2.imwrite(nomImg, img)    

    # --------SEPARATION DES CARACTERES----------
    img_caractere = [[] for i in range(len(img_ligne))]
    coord_caracteres_ligne = [[] for i in range(len(img_ligne))]

    for i, ligne in enumerate(img_ligne):
        
        # je recupere les dimensions de l'image de la ligne
        height_line, _ = ligne.shape
        # on inverse les couleurs
        _, img_threshold = cv2.threshold(ligne, 125, 255, cv2.THRESH_BINARY_INV)

        # on cree l'histogramme de la ligne
        
        vertical_pixel_sum = np.sum(img_threshold, axis=0)
        myprojectionx = vertical_pixel_sum / 255

        # on calcule les coordonnees de la delimitaion des caracteres
        coord_caracteres = coordonneeCaractere(myprojectionx)

        coord_caracteres_ligne[i].append(coord_caracteres)

    # on segmente l'image pour separer les caracteres
        for j in range(len(coord_caracteres)):
            _, uncaractereBI = cv2.threshold(ligne[0:height_line, coord_caracteres[j][0]:coord_caracteres[j][1]], 125, 255, cv2.THRESH_BINARY)
            img_caractere[i].append(uncaractereBI)
        
    # on vérifie la pertinance des images de lettre générées
    img_caractere = trillageCaractere(img_caractere)

    # on donne une dimension de 128x128 aux caractères 
    for i in range (len(img_caractere)):
        for j in range (len(img_caractere[i])):
            img_caractere[i][j] = cv2.resize(src=img_caractere[i][j], dsize=(128, 128), interpolation=cv2.INTER_NEAREST_EXACT)
            _, img_caractere[i][j] = cv2.threshold(img_caractere[i][j], 127, 255, cv2.THRESH_BINARY)
    return coord_ligne, img_ligne, coord_caracteres_ligne, img_caractere

# imgTest = cv2.imread("aSegmenterPropre.png")
# _, _, _, test = segmentation(imgTest)


# try: 
# 	os.mkdir("imageSegmentee") # le dossier n'existe pas
# except OSError as error: 
	
# 	#le dossier existe déjà, on détruit les fichiers
# 	for filename in os.listdir("imageSegmentee"):
# 		os.remove("imageSegmentee/" + filename)

# line = 0
# column = 0
# for ligne in test:
# 	column = 0
# 	for image in ligne:
# 		cv2.imwrite("imageSegmentee/"+str(line)+"_" + str(column) + "lettre.png", image)
# 		column +=1
# 	line += 1
	






