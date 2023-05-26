import tkinter
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os

from widgetUtiles import *



class FenetrePhoto(ctk.CTkToplevel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.geometry("1000x700")

		self.labelPhoto = LabelPhoto(self)
		self.labelPhoto.grid(row=0, column=0, padx=20, pady=20)

		self.buttonScreenShot = ButtonApp(self, "Prendre une Photo", self.enregistrerPhoto )
		self.buttonScreenShot.grid(row=1, column=0, padx= 20, pady=20)

		self.bind('<Escape>', lambda e: self.quitter())
		self.bind('<space>', lambda e: self.enregistrerPhoto())

		self.protocol("WM_DELETE_WINDOW", self.quitter) #comme quand on appui sur échap mais c'est quand on ferme la fenetre

		# Define a video capture object
		self.vid = cv2.VideoCapture(0)
		
		# Declare the width and height in variables
		width, height = 1920, 1080
		
		# Set the width and height
		self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		self.afficherImage()
		
	def quitter(self, destructionPhoto=True):
		#c'est pour faire le release avant de faire destroy #TechniqueDeDev

		if destructionPhoto: # si on fait échap, on veut pas que la photo soit mise dans le label photo de la fenetre principale
			if os.path.exists('images/imageOriginale.png'):
				os.remove('images/imageOriginale.png')
		

		self.vid.release()
		self.destroy()


	def enregistrerPhoto(self):
		#faut prendre une photo, l'enregistrer et prévenir la fenetre principale que c'est bon
		print("flash")
		photo_image = self.prend1Image()
		cv2.imwrite('images/imageOriginale.png', photo_image)
		self.quitter(False)
	
	def prend1Image(self):
		# Capture the video frame by frame

		_, frame = self.vid.read()
	
		# Convert image from one color space to other
		opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	
		return opencv_image

	
	def afficherImage(self):
		
		self.labelPhoto.actualiserImage(self.prend1Image())

		# execute afficherImage après 15ms
		self.labelPhoto.after(15, self.afficherImage)
	