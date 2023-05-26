import customtkinter as ctk
from PIL import Image, ImageDraw


class ButtonApp(ctk.CTkButton):
	def __init__(self, master, text, command):
		super().__init__(master = master, text=text, command=command, font=("font1", 25))




class LabelPhoto(ctk.CTkLabel):
	def __init__(self, master, size=(840,480)):
		super().__init__(master = master, text = "")
		self.tailleCadre = size

	
	def actualiserImage(self, image):

		#image est une openCV
		imagePIL = Image.fromarray(image)

		imageCTk = ctk.CTkImage(light_image=imagePIL, size=self.tailleCadre)
		# Displaying photoimage in the label
		self.photo_image = imageCTk
	
		# Configure image in the label
		self.configure(image=imageCTk)



	# def creerImageTransparente(self):

	# 	img = Image.new('RGBA', self.tailleCadre)
	# 	imageCTk = ctk.CTkImage(light_image=img, size=self.tailleCadre)
	# 	self.photo_image = imageCTk
	# 	self.configure(image=imageCTk)



	# def creerLigne(self, img, px0, px1, py0, py1, largeur=840, hauteur=480):
		
	# 	x0, x1 = int(px0 * largeur), int(px1 * largeur)
	# 	y0, y1 = int(py0 * hauteur), int(py1 * hauteur)
	# 	img = Image.new('RGBA', (largeur, hauteur))
	# 	draw = ImageDraw.Draw(img)
	# 	draw.line([(0,hauteur - y0), (largeur, hauteur - y0)], fill ="black", width = 2)
	# 	draw.line([(0,hauteur - y1), (largeur, hauteur - y1)], fill ="black", width = 2)
	# 	draw.line([(x0,0), (x0, hauteur)], fill ="black", width = 2)
	# 	draw.line([(x1,0), (x1, hauteur)], fill ="black", width = 2)

	# 	imageCTk = ctk.CTkImage(light_image=img, size=self.tailleCadre)
	# 	self.photo_image = imageCTk
	# 	self.configure(image = imageCTk)



#classe prise de la page https://customtkinter.tomschimansky.com/tutorial/spinbox
class Spinbox(ctk.CTkFrame):
	def __init__(self, *args,
				 width: int = 150,
				 height: int = 48,
				 step_size = 1,
				 min = 0,
				 max = 100,
				 command = None,
				 **kwargs):
		super().__init__(*args, width=width, height=height, **kwargs)

		self.step_size = step_size
		self.command = command
		self.max=max
		self.min=min

		self.configure(fg_color=("gray78", "gray28"))  # set frame color

		self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
		self.grid_columnconfigure(1, weight=1)  # entry expands

		self.subtract_button = ctk.CTkButton(self, text="-", width=height-6, height=height-6,
													   command=self.subtract_button_callback)
		self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

		self.entry = ctk.CTkEntry(self, width=width-(2*height), height=height-6, border_width=0, font=("font1", 25), state="disabled")
		self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

		self.add_button = ctk.CTkButton(self, text="+", width=height-6, height=height-6,
												  command=self.add_button_callback)
		self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

		# default value
		self.entry.insert(0, "0")

	def add_button_callback(self):
		
		if int(self.entry.get()) == self.max:
			return
		try:
			value = int(self.entry.get()) + self.step_size
			self.entry.configure(state="normal")
			self.entry.delete(0, "end")
			self.entry.insert(0, value)
			self.entry.configure(state="disabled")
		except ValueError:
			return
		if self.command is not None:
			self.command(1)

	def subtract_button_callback(self):
		if int(self.entry.get()) == self.min:
			return
		try:
			value = int(self.entry.get()) - self.step_size
			self.entry.configure(state="normal")
			self.entry.delete(0, "end")
			self.entry.insert(0, value)
			self.entry.configure(state="disabled")
		except ValueError:
			return
		if self.command is not None:
			self.command(-1)

	def get(self):
		try:
			return int(self.entry.get())
		except ValueError:
			self.set(0)
			return None

	def set(self, value: int):
		self.entry.configure(state="normal")
		self.entry.delete(0, "end")
		self.entry.insert(0, str(int(value)))
		self.entry.configure(state="disabled")