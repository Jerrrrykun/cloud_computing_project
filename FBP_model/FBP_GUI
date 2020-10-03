import tkinter as tk  # Construct GUI for 
from tkinter import filedialog, messagebox
from scipy.misc import imresize  # For recieve new photos
from keras.preprocessing.image import load_img, img_to_array  # Read pics
from keras.models import load_model
from PIL import Image, ImageTk  # Show pics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # U could annotate this if ur computer do not have an extension on CPU


#######################################################################################################################
class My_GUI(tk.Tk):

    IMAGE_SUFFIX = ('.jpg', '.jpeg', '.png')  # The allowed files' suffix 

    def __init__(self):
        super().__init__()  # Initialize the class
        self.title('Face Beauty Scoring App')
        self.ini_ui()
        self.image = None
        self.image_for_model = None
        self.img_label = tk.Label(master=self)  # Instantialization
        self.score_label = tk.Label(master=self)  # Show the score
        self.img_height, self.img_width, self.channels = 350, 350, 3

    def open_image(self, image_path=None):  # Open and show the image
        if image_path is None:  # No file input
            image_path = filedialog.askopenfilename(title='Open Image File')  # Ask something
            if image_path == '':
                return  # Just for opening the image
            elif not image_path.lower().endswith(self.IMAGE_SUFFIX):  # No allowed image format
                messagebox.showerror('Error', 'Wrong Image Format!!!!') 
                return
        self.image_for_model = load_img(image_path)
        self.image = Image.open(image_path)  # Opem the image
        self.w = self.image.size[0]
        self.h = self.image.size[1]
        if self.w > 1000 or self.h > 1000:  
            self.image = self.image.resize((300, 400), Image.ANTIALIAS) 
        self.image = ImageTk.PhotoImage(self.image)
        self.img_label.configure(image=self.image)
        self.img_label.image = self.image 
        self.img_label.pack()

    def model_prediction(self):  
        self.image_for_model = imresize(self.image_for_model, size=(self.img_height, self.img_width))
        self.test_x = img_to_array(self.image_for_model).reshape(self.img_height, self.img_width, self.channels)
        self.test_x /= 255.0
        self.test_x = self.test_x.reshape((1,) + self.test_x.shape)  
        self.model = load_model('25-0.097(Xception with dropout(0.2)).h5')  # You should put your trained model here
        self.predicted = self.model.predict(self.test_x)
        self.score_label.configure(text='Your beauty score is: {:.3} out of 5'.format(self.predicted[0][0]))
        self.score_label.text = 'Your beauty score is: {:.3} out of 5'.format(self.predicted[0][0])  
        self.score_label.pack()

    def ini_ui(self):  # Buttons
        self.btn = tk.Button(self, text='Start Face Beauty Scoring', font=('Helvetica', '20'))
        self.btn.pack(padx=200, pady=30)
        self.btn.config(command=self.model_prediction)

        self.open_image_button = tk.Button(self, text='Open the Image', font=('Helvetica', '15'), command=self.open_image)
        self.open_image_button.pack()

        self.version_button = tk.Button(self, text='Project Information', font=('Helvetica', '15'),
                                        command=self.show_project_information)
        self.version_button.pack()

    def show_project_information(self):
        messagebox.showinfo('Project Information', 'Version 1.0\nDeep Learning Final Project\nAuthor: Xu Zhikun')  # Show the version info


#######################################################################################################################
if __name__ == '__main__':
    app = My_GUI()
    app.mainloop()
