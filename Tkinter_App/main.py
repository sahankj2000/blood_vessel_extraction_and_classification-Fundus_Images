from tkinter import *
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np

root = Tk()

selected_path = ''

def select_image():
    filename = filedialog.askopenfilename(filetypes=[('JPEG','.jpg .jpeg .JPG')]) #filetypes=[('JPEG','.jpg .jpeg')]
    img = Image.open(filename)
    global selected_path
    selected_path = filename
    img = img.resize((512,512))
    img = ImageTk.PhotoImage(img)
    input_label.configure(image=img)
    input_label.image = img

def extract():
    if selected_path == '':
        messagebox.showerror('No image selected','Please select an image to proceed')
    else:
        predict()
        img = ImageTk.PhotoImage(Image.open('extracted/predicted.jpg'))
        output_label.configure(image=img)
        output_label.image = img


encoder_input = keras.Input(shape=(512,512,3),name='img')
l1 = keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu',
                        activity_regularizer = regularizers.l1(10e-10))(encoder_input)
#l2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(l1)
l2 = keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu',
            activity_regularizer = regularizers.l1(10e-10))(l1)
#l4 = keras.layers.UpSampling2D()(l3)
l3 = keras.layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu',
            activity_regularizer = regularizers.l1(10e-10))(l2)

l4 = keras.layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu',
            activity_regularizer = regularizers.l1(10e-10))(l3)

decoder_output = keras.layers.Conv2D(1, (3, 3), padding = 'same', activation = 'relu',
            activity_regularizer = regularizers.l1(10e-10))(l4)

opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)

autoencoder = keras.Model(encoder_input,decoder_output,name='autoencoder')
autoencoder.summary()
autoencoder.load_weights('ckpt/cp.ckpt')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # # The fifth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()
model.load_weights('ckpt1/cp.ckpt')


def predict():
    image = Image.open(selected_path)
    image = image.resize((512,512))
    image = np.array(image)
    image = image/255.0
    image = image.reshape(-1,512,512,3)
    extracted = autoencoder.predict(image)[0]
    classify = model.predict(image.reshape(-1,512,512,3))[0]
    if classify[0] >= 0.5:
        result = "DR affected"
    else:
        result = "Healthy"
    prediction_label.configure(text='Classification : '+result)
    plt.imsave('extracted/predicted.jpg',extracted.reshape(512,512),cmap='gray')


button_font = tkfont.Font(size=16)
pred_font = tkfont.Font(size=22)

select_button = Button(root,text='Select Image',command=select_image,width=37,font=button_font)
select_button.place(x=20,y=20)

extract_button = Button(root,text='Extract',command=extract,width=37,font=button_font)
extract_button.place(x=552,y=20)

input_frame = Frame(root,width=512,height=512)
input_frame.place(x=20,y=80)
input_label = Label(input_frame)
input_label.pack()

output_frame = Frame(root,width=512,height=512)
output_frame.place(x=552,y=80)
output_label = Label(output_frame)
output_label.pack()

prediction_label = Label(root,text='Classification: ',font=pred_font)
prediction_label.place(x=400,y=620)


root.title("Blood Vessel Extraction and Classification")
root.geometry("1084x680")
root.resizable(False,False)
root.mainloop()
