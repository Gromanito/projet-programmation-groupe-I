import tkinter
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os

from widgetUtiles import *
from FrameTraitementImage import FrameTraitementImage



class FenetreModifPhoto(ctk.CTkToplevel):


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.geometry("1920x1080")

		self.bind('<Escape>', lambda e : self.destroy())

		

		self.frameTraitementImage = FrameTraitementImage(self)
		self.frameTraitementImage.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

		

		self.buttonReprendrePhotoOriginale = ButtonApp(self, text="Reprendre photo originale", command=self.photo_originale)
		self.buttonValider = ButtonApp(self, text="Valider", command=self.valider)


		self.buttonReprendrePhotoOriginale.grid(row=1, column=0, padx=20, pady=20)
		self.buttonValider.grid(row=1, column=1, padx=20, pady=20)

		



	def valider(self):
		self.frameTraitementImage.enregistrerModif()
		self.destroy()

	

	def photo_originale(self):
		self.frameTraitementImage.prendrePhotoOriginale()

	
	
	
	