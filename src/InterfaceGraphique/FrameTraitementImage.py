import tkinter
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import math

from widgetUtiles import *
from TraitementImage import *


	
"""servira à choisir la binarisation et 
à créer des nouvelles frames pour rogner et pour rotater l'image"""
class FrameTraitementImage(ctk.CTkFrame):
	# ce sera à l'utilisateur de choisir la binarisation qu'il veut
	dicoTypeSeuil = {
	"SEUIL_BASIQUE" : 0,
	"SEUIL_ADAPTATIF" : 1,
	"SEUIL_ADAPTATIF_GAUSSIEN" : 2,
	"SEUIL_SAUVOLA": 3,
	"DEFAUT" : 4,
	"0" : "SEUIL_BASIQUE" ,
	"1" : "SEUIL_ADAPTATIF",
	"2" : "SEUIL_ADAPTATIF_GAUSSIEN",
	"3" : "SEUIL_SAUVOLA",
	"4" : "DEFAUT"
	}

	cheminImageModif="images/imageModif.png"
	cheminImageBin="images/imageBin.png"

	paramTypeBinarisation = 4
	paramSeuil = 128
	enleverFond = False
	conserverBoundingBox = False
	angleOpti = 0

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		


		self.typeBinarisation = FrameTraitementImage.paramTypeBinarisation
		self.seuilBin = FrameTraitementImage.paramSeuil
		self.angleRotation = 0

		self.imageModif = cv2.imread(FrameTraitementImage.cheminImageModif)


		

		self.frameDuBas = ctk.CTkFrame(self) #contiendra les 3 compartiments pour rogner, pivoter et choisir la binarisation
		self.frameDuBas.grid(row=1, column=0, columnspan=2)

		frameRognage = ctk.CTkFrame(self.frameDuBas)
		frameRognage.grid(row=1, column=0, padx=40, pady=20)

		frameRotation = ctk.CTkFrame(self.frameDuBas)
		frameRotation.grid(row=1, column=1, padx=40, pady=20, sticky="ns")
		
		frameBinarisation = ctk.CTkFrame(self.frameDuBas)
		frameBinarisation.grid(row=1, column=2, padx=40, pady=20)


		frameGauche = ctk.CTkFrame(self)
		frameGauche.grid(row=0,column=0, padx=20, pady=20)

		frameDroite = ctk.CTkFrame(self)
		frameDroite.grid(row=0,column=1, padx=20, pady=20)
		
		
		#label qui contiendra la photo (ptre modifiée)
		self.canvasPhotoModif = tkinter.Canvas(frameGauche, width=840, height=480)
		self.canvasPhotoModif.grid(row=0, column=0, padx=20, pady=20)
		self.imageTKPourMemoire = None
		

		self.labelPhotoBin = LabelPhoto(frameDroite)
		self.labelPhotoBin.grid(row=0, column=0, padx=20, pady=20)


		labelRenduPourYolo = ctk.CTkLabel(frameDroite, text="Rendu binaire (utilisé pour YOLO)", font=("font1", 14))
		labelRenduPourYolo.grid(row=1, column=0)



		


		
		#----------  Création compartiment rotation -----

		self.spinbox = Spinbox(frameRotation, min=-180, max=180, command=self.rotation)
		self.spinbox.grid(row=0, padx=50, pady=30)
		self.spinbox.set(0)


		if self.typeBinarisation == 4:
			FrameTraitementImage.conserverBoundingBox = True
			self.angleRotation = FrameTraitementImage.angleOpti
			self.spinbox.set(self.angleRotation)
			self.imageModif = rotation(self.imageModif, self.angleRotation, True)
			cv2.imwrite(FrameTraitementImage.cheminImageModif , self.imageModif)




		self.buttonBoundingBox = ctk.CTkRadioButton(frameRotation, text="Conserver Toute\nL'image", value=0 ,font=("font1", 18), command=self.rotaBoundBox)
		if FrameTraitementImage.conserverBoundingBox == True:
			self.buttonBoundingBox.select()
		self.buttonBoundingBox.grid(row=1, padx=50, pady=30)




		#----------  Création compartiment binarisation -----

		self.choixTypeBin = ctk.CTkComboBox(frameBinarisation, values=["SEUIL_BASIQUE","SEUIL_ADAPTATIF","SEUIL_ADAPTATIF_GAUSSIEN", "SEUIL_SAUVOLA", "DEFAUT"], font=("font1", 15), dropdown_font=("font1", 15), command=self.changeTypeBin)
		self.choixTypeBin.set(FrameTraitementImage.dicoTypeSeuil[str(self.typeBinarisation)])
		self.choixTypeBin.grid(row=0, column=0, padx=60, pady=30)

		self.buttonGarderQueFeuille = ctk.CTkRadioButton(frameBinarisation, text="Enlever fond", value=0, font=("font1", 18) ,command=self.garderQueFeuille)
		if FrameTraitementImage.enleverFond == True:
			self.buttonGarderQueFeuille.select()

		self.buttonGarderQueFeuille.grid(row=1, column=0, padx=60, pady=40)


		self.choixSeuil = ctk.CTkSlider(frameBinarisation, orientation="horizontal", from_=0, to=255, number_of_steps=256, width=300, height=10, command=self.changeSeuil)
		self.choixSeuil.grid(row=0, column=1, padx=60, pady=30)
		self.choixSeuil.set(self.seuilBin)

		boutonAppliquerFiltreMedian = ButtonApp(frameBinarisation, text="Filtre\Median", command=self.appliquerFiltreMedian)
		boutonAppliquerFiltreMedian.grid(row=1, column=1, padx=60, pady=40)


		self.changeTypeBin(0)
		self.actualiserImage()



		#----------  Création compartiment rognage -----
		hauteur, largeur, _= self.imageModif.shape

		
		self.canvasRognage = ctk.CTkCanvas(frameRognage, width = 210, height=120)
		self.sliderX0 = ctk.CTkSlider(frameRognage, orientation="horizontal", from_=0, to=largeur, number_of_steps=210, width=210, height=10, command=self.dessinerLigne)
		self.sliderX1 = ctk.CTkSlider(frameRognage, orientation="horizontal", from_=0, to=largeur, number_of_steps=210, width=210, height=10, command=self.dessinerLigne)
		self.sliderY0 = ctk.CTkSlider(frameRognage, orientation="vertical"  , from_=0, to=hauteur, number_of_steps=120, width=10, height=120, command=self.dessinerLigne)
		self.sliderY1 = ctk.CTkSlider(frameRognage, orientation="vertical"  , from_=0, to=hauteur, number_of_steps=120, width=10, height=120, command=self.dessinerLigne)

		self.rognageParDefaut()

		boutonRogner = ButtonApp(frameRognage, text="Rogner", command=self.rogner)

		self.canvasRognage.grid(row=1, column=1)
		self.sliderX0.grid(row=0, column=1, padx=15, pady=15)
		self.sliderX1.grid(row=2, column=1, padx=15, pady=15)
		self.sliderY0.grid(row=1, column=0, padx=15, pady=15)
		self.sliderY1.grid(row=1, column=2, padx=15, pady=15)

		boutonRogner.grid(row=0,column=3, rowspan=3, padx=20, pady=20)



	def dessinerLigne(self, val):
		hauteur, largeur, _= self.imageModif.shape

		x0, x1, y0, y1 = int( self.sliderX0.get() ), int( self.sliderX1.get() ), int( self.sliderY0.get() ), int( self.sliderY1.get() )
		# x0, x1, y0, y1  sont les valeurs max de l'image (qui varient puisqu'on rogne) 
		# et qu'il faut adapter aux cadres, qui eux bougent pas

		px0, px1, py0, py1 = (x0/largeur), (x1/largeur), (y0/hauteur), (y1/hauteur)

		
		#self.canvasPhotoModif.creerLigne(px0, px1, py0, py1)

		#210 la valeur max que x0 et x1 peuvent prendre
		



		#on efface le canvas
		self.canvasRognage.delete("all")
		self.canvasRognage.create_line(0,      120-int(120 * py0),       210,     120-int(120 * py0))
		self.canvasRognage.create_line(0,      120-int(120 * py1),       210,     120-int(120 * py1))
		self.canvasRognage.create_line(int(210 * px0),      	0,   	int(210 * px0),          120)
		self.canvasRognage.create_line(int(210 * px1),			0, 		int(210 * px1), 		 120)


		x0, x1 = int(840 * px0), int(840 * px1)
		y0, y1 = int(480 * py0), int(480 * py1)


		self.canvasPhotoModif.delete("lignes")
		self.canvasPhotoModif.create_line(0,   480 - y0,    840,    480 - y0, tags="lignes")
		self.canvasPhotoModif.create_line(0, 480 - y1, 840, 480 - y1, tags="lignes")
		self.canvasPhotoModif.create_line(x0, 0, x0, 480, tags="lignes")
		self.canvasPhotoModif.create_line(x1, 0, x1, 480, tags="lignes")





	def appliquerFiltreMedian(self):
		self.imageModif = cv.medianBlur(cv2.imread(FrameTraitementImage.cheminImageModif), 3)
		cv2.imwrite(FrameTraitementImage.cheminImageModif, self.imageModif)
		self.actualiserImage()
	
	# def appliquerFiltreGaussien(self):
	# 	self.imageModif = cv.GaussianBlur(grayImage, (5,5), 0)


	def rogner(self):

		if self.angleRotation != 0:
			self.imageModif = rotation(self.imageModif, self.angleRotation, FrameTraitementImage.conserverBoundingBox)
		

		hauteur, _, _= self.imageModif.shape
		

		x0, x1, y0, y1 = int (self.sliderX0.get()), int(self.sliderX1.get()), int(self.sliderY0.get()), int(self.sliderY1.get())

		y0 = hauteur-y0
		y1 = hauteur-y1
		if x0 >= x1:
			x1, x0 = x0, x1
		if y0 >= y1:
			y1, y0 = y0, y1
		

		self.canvasRognage.delete("all") #on efface le canvas
		self.canvasPhotoModif.delete("all")
		
		self.imageModif= (self.imageModif[y0:y1, x0:x1])
		if FrameTraitementImage.dicoTypeSeuil[self.choixTypeBin.get()] == 4:
			self.imageBin = self.imageBin[y0:y1, x0:x1]

		cv2.imwrite(FrameTraitementImage.cheminImageModif, self.imageModif)
		cv2.imwrite(FrameTraitementImage.cheminImageBin, self.imageBin)
		self.rognageParDefaut() #on remet les sliders comme il faut avec les bonnes valeurs
		self.spinbox.set(0) #on réinitialise la rotation au cas où
		self.angleRotation = 0
		self.actualiserImage()
	
	

	def changeSeuil(self, valeur):
		self.seuilBin=valeur
		self.actualiserImage()
	

	def garderQueFeuille(self):
		FrameTraitementImage.enleverFond = not FrameTraitementImage.enleverFond
		self.actualiserImage()
		if FrameTraitementImage.enleverFond == True:
			self.buttonGarderQueFeuille.select()
		else:
			self.buttonGarderQueFeuille.deselect()
			



	def changeTypeBin(self, event):

		self.typeBinarisation = FrameTraitementImage.dicoTypeSeuil[self.choixTypeBin.get()]
		if self.typeBinarisation == 0:
			self.choixSeuil.grid(row=0, column=1)
		else:
			self.choixSeuil.grid_forget()

		self.actualiserImage()



	def rotation(self, nombre):
		self.angleRotation += nombre
		
		self.actualiserImage()

	
	def rotaBoundBox(self):
		FrameTraitementImage.conserverBoundingBox = not FrameTraitementImage.conserverBoundingBox
		if FrameTraitementImage.conserverBoundingBox == True:
			self.buttonBoundingBox.select()
		else:
			self.buttonBoundingBox.deselect()




	def enregistrerModif(self): #dès qu'on change, on enregistre (on "commit") les modifications faites
		cv2.imwrite(FrameTraitementImage.cheminImageModif, self.imageModif)
		cv2.imwrite(FrameTraitementImage.cheminImageBin, self.imageBin)
		FrameTraitementImage.paramSeuil = self.seuilBin
		FrameTraitementImage.paramTypeBinarisation = self.typeBinarisation

	def prendrePhotoOriginale(self):
		#faut aussi remettre la rotation à 0, et réinitialiser le rognage
		self.spinbox.set(0) #on réinitialise la rotation au cas où
		self.angleRotation = 0
		cv.imwrite(FrameTraitementImage.cheminImageModif, cv2.imread("images/imageOriginale.png"))
		if FrameTraitementImage.dicoTypeSeuil[self.choixTypeBin.get()] == 4:
			self.imageModif = cv2.imread("images/imageOriginale.png")
			self.angleRotation = FrameTraitementImage.angleOpti
			self.imageBin = cv2.imread("images/imageBinDefaut.png")
			cv2.imwrite(FrameTraitementImage.cheminImageBin, self.imageBin)
			self.spinbox.set(self.angleRotation) 
			
			FrameTraitementImage.conserverBoundingBox = True
			self.imageModif = rotation(self.imageModif, self.angleRotation, FrameTraitementImage.conserverBoundingBox)
		self.actualiserImage()
		self.rognageParDefaut()
		self.enregistrerModif()
	
	def rognageParDefaut(self):
		hauteur, largeur, _= self.imageModif.shape
		self.sliderX1.configure(to=largeur)
		self.sliderX0.configure(to=largeur)
		self.sliderY0.configure(to=hauteur)
		self.sliderY1.configure(to=hauteur)

		self.sliderX1.set(largeur)
		self.sliderX0.set(0)
		self.sliderY0.set(hauteur)
		self.sliderY1.set(0)


	# def outilsRognage(self):
	# 	return

	
	def actualiserImage(self): # montre la nouvelle image modifiée (et l'image binarisée)

		if FrameTraitementImage.dicoTypeSeuil[self.choixTypeBin.get()] != 4: #c'est la binarisation compliqué, on recalcule pas
		
			if self.angleRotation != 0: #si y a une rotation, faut binariser avant de faire la rotation, sinon on binarise aussi les contours et ça fait des trucs moches
				imageSansRota = cv2.imread(FrameTraitementImage.cheminImageModif)  
				self.imageBin = rotation( binarisation(imageSansRota, self.typeBinarisation, self.seuilBin, FrameTraitementImage.enleverFond) , self.angleRotation, FrameTraitementImage.conserverBoundingBox)
				self.imageBin = binarisation(self.imageBin, Seuil=127, dejaNDG=True)
				self.imageModif = rotation( imageSansRota  , self.angleRotation, FrameTraitementImage.conserverBoundingBox)
			
			else:
				self.imageModif = cv2.imread(FrameTraitementImage.cheminImageModif)
				self.imageBin = binarisation(self.imageModif, self.typeBinarisation, self.seuilBin, FrameTraitementImage.enleverFond)

		else:
			if self.angleRotation != 0:
				self.imageModif = rotation( cv2.imread(FrameTraitementImage.cheminImageModif)  , self.angleRotation, FrameTraitementImage.conserverBoundingBox)
				self.imageBin = cv2.imread(FrameTraitementImage.cheminImageBin, cv2.IMREAD_GRAYSCALE)
				self.imageBin = binarisation (rotation(self.imageBin, self.angleRotation, FrameTraitementImage.conserverBoundingBox ), typeBinarisation=0 , Seuil=127, dejaNDG=True)
			else:
				self.imageBin = cv2.imread(FrameTraitementImage.cheminImageBin, cv2.IMREAD_GRAYSCALE)
				self.imageModif = cv2.imread(FrameTraitementImage.cheminImageModif)



		imagePIL= Image.fromarray(self.imageModif)
		imagePIL = imagePIL.resize((840,480), Image.ANTIALIAS)
		imageTK = ImageTk.PhotoImage(imagePIL)
		self.imageTKPourMemoire=imageTK
		
		self.canvasPhotoModif.delete("image")
		self.canvasPhotoModif.create_image(0, 0, anchor="nw", image=imageTK, tags="image")
		self.canvasPhotoModif.image= self.imageTKPourMemoire


		self.labelPhotoBin.actualiserImage(self.imageBin)




	
	
	
	