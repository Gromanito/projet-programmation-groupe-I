import tkinter
import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

from widgetUtiles import *
from fenetrePhoto import FenetrePhoto
from fenetreModifPhoto import FenetreModifPhoto
from segmente import *
from TraitementImage import *
import FrameTraitementImage




ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


try:
	os.mkdir("images") # le dossier n'existe pas
except OSError as error: 
	pass 


class FenetrePrincipale(ctk.CTk):
	def __init__(self):
		super().__init__()
		self.bind('<Escape>', lambda e : self.destroy())

		self.topLevel = None # c'est pour avoir un pointeur sur la future nouvelle fenetre créée
		
		self.labelRenduPourYolo = None
		self.frameDuBas = None
		self.frameTexteReconnu=None


		self.geometry("700")
		self.title("Titre jsp")
		self.minsize(300, 200)

		# create 2x2 grid system
		# grid_xconfigure : ( numero ou tuples de(s) x concerné(s) ,     poids de ce(s) x concerné(s)  )
			# self.grid_rowconfigure(0, weight=5)
			# self.grid_rowconfigure(1, weight=4)

		self.recommencer()


	def recommencer(self):
		self.destroyWidget(self.topLevel)
		self.destroyWidget(self.labelRenduPourYolo)
		self.destroyWidget(self.frameDuBas)
		self.destroyWidget(self.frameTexteReconnu)

		self.frameDuBas = ctk.CTkFrame(self)
		self.frameDuBas.grid(row=1, column=0, padx=250, pady=75)


		self.labelExplication = ctk.CTkLabel(master=self, text="Prenez une photo ou sélectionnez un fichier", font=("font1", 35))
		buttonPrendrePhoto = ButtonApp(master=self.frameDuBas, text="Prendre une photo", command=self.button_prendre_photo)
		buttonSelectionnerFichier = ButtonApp(master = self.frameDuBas, text="Selectionner un fichier", command=self.browseFiles)
		
		self.labelExplication.grid(row=0, column=0, padx=250, pady=150)
		buttonPrendrePhoto.grid(row=0, column=0, padx=50, pady=50)
		buttonSelectionnerFichier.grid(row=0, column=1, padx=50, pady=50)
		


	def afficherPhotoAReconnaitre(self):
		self.destroyWidget(self.labelExplication)
		self.destroyWidget(self.frameDuBas)
		self.destroyWidget(self.frameTexteReconnu)



		self.frameDuBas = ctk.CTkFrame(self)
		self.frameDuBas.grid(row=1, column=0, padx=20, pady=25)
		

		self.labelRenduPourYolo = LabelPhoto(self, (1280,720))
		self.labelRenduPourYolo.grid(row=0, column=0, padx=10, pady=10)
		
		imgAppPhoto = cv2.imread("images/imageOriginale.png")
		cv2.imwrite("images/imageModif.png", imgAppPhoto)
		angleOpti, imageBinDefaut = appliquerBinarisationParDefaut(imgAppPhoto)
		cv2.imwrite("images/imageBinDefaut.png", imageBinDefaut)
		cv2.imwrite("images/imageBin.png", imageBinDefaut)
		FrameTraitementImage.angleOpti = angleOpti


		self.labelRenduPourYolo.actualiserImage(cv2.imread("images/imageBin.png"))


		
		buttonModifImage = ButtonApp(self.frameDuBas, "Modifier", self.button_modifier_image)
		buttonModifImage.grid(row=0, column=0, padx=10,pady=10)

		
		buttonRecommencer = ButtonApp(self.frameDuBas, "Recommencer", self.recommencer)
		buttonRecommencer.grid(row=0, column=1, padx=10, pady=10)

		buttonLancerAnalyse = ButtonApp(self.frameDuBas, "Lancer Reconnaissance Manuscrite", self.analyse)
		buttonLancerAnalyse.grid(row=1, column=0, columnspan=2, padx=10, pady=10)


	

	def analyse(self):
		self.destroyWidget(self.frameTexteReconnu)

		cheminCWD = os.getcwd() # yolo comprend pas les chemins relatifs ...

		model = YOLO(cheminCWD+"/src/pourYOLO/entrainementsYolo/bestUpperSansBord.pt")

		imgASegmenter = cv2.imread("images/imageBin.png")

		_, _, _, imagettes = segmentation(imgASegmenter)

		ecritImagetteDansFichier(imagettes)


		lignes=[]
		for dossier in sorted(os.listdir("images/imageSegmentee")):
			
			rendPretPourYolo("images/imageSegmentee"+'/' + dossier)

			results = model.predict(source=cheminCWD+'/images/imageSegmentee/'+ dossier, verbose = False)

			
			listeReconnus = []
			for r in results:
				array = r.probs.numpy()
				maxConfiance=0
				indice=-1
				for i, proba in enumerate(array):
					if proba > maxConfiance:
						indice, maxConfiance = i, proba
				listeReconnus.append(model.names[indice])
			print("".join(listeReconnus))
			lignes.append("".join(listeReconnus))
		
		#lignes contient le texte reconnu


		self.frameTexteReconnu = ctk.CTkFrame(self)
		self.frameTexteReconnu.grid(row=2, column=0, pady=20)

		labelTexteReconnu = ctk.CTkLabel(self.frameTexteReconnu, text="Texte reconnu : ", font=("font1", 30))
		labelTexteReconnu.grid(row=0, column=0, padx=100, pady=5)

		labelTexteReconnuYolo = ctk.CTkLabel(self.frameTexteReconnu, text="\n".join(lignes), font=("font1", 20))
		labelTexteReconnuYolo.grid(row=0, column=1, padx=10, pady=5)
		
		
		

		
	
	def button_modifier_image(self): #ouvre une fenetre pour modifier l'image
		if self.topLevel is None or not self.topLevel.winfo_exists(): # on regarde qu'aucune autre fenetre existe
			self.topLevel = FenetreModifPhoto(self)  # create window if its None or destroyed
			self.wait_visibility(self.topLevel)
			self.topLevel.grab_set() #redirige les événements vers la fenetre qui vient d'etre créée
			self.wait_window(self.topLevel)
			self.topLevel = None #pour détruire la fenetre (normalement)
			self.grab_set()

			self.labelRenduPourYolo.actualiserImage(cv2.imread("images/imageBin.png"))


		else:
			self.topLevel.focus()


	def button_prendre_photo(self): #ouvre une fenetre pour prendre une photo

		
		if self.topLevel is None or not self.topLevel.winfo_exists(): # on regarde qu'aucune autre fenetre existe
			self.topLevel = FenetrePhoto(self)  # create window if its None or destroyed
			self.wait_visibility(self.topLevel)
			self.topLevel.grab_set() #redirige les événements vers la fenetre qui vient d'etre créée
			self.wait_window(self.topLevel)
			self.topLevel = None #pour détruire la fenetre (normalement)
			self.grab_set()

			if os.path.exists('images/imageOriginale.png'):
				self.afficherPhotoAReconnaitre()
			else:
				self.recommencer()

		else:
			self.topLevel.focus()



	def browseFiles(self):
		filename = filedialog.askopenfilename(initialdir = "/home",
													title = "Select a File",
													filetypes = (("Jpeg files", "*.jpg*"),("Png files","*.png")))
		if filename != () or filename == "":
			print(filename)
			cv2.imwrite("images/imageOriginale.png",cv2.imread(filename))
			self.afficherPhotoAReconnaitre()
			#afficher la photo, modifier la fenetre bref faut reflechir un peu
			#faut aussi fermer la toplevel si elle est ouverte
		else:
			print("Aucun fichier sélectionné")
			# on fait rien

		
	
	def destroyWidget(self, widget):
		""" regarde si le widget existe, et le destroy  """
		if not (widget is None or not widget.winfo_exists()):
			widget.destroy()

      
        

if __name__ == "__main__":
    app = FenetrePrincipale()
    app.mainloop()

