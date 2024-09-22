import tkinter as tk
import numpy as np
import tensorflow as tf
from tkinter import Button, Canvas, Label
from PIL import Image, ImageDraw, ImageOps

# Load the saved model
model = tf.keras.models.load_model('mnist_model.h5')

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        # Create a Canvas widget for drawing
        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack(padx=10, pady=10)

        # Add a clear button
        clear_button = Button(root, text="Clear", command=self.clear_canvas)
        clear_button.pack(side='left', padx=10)

        # Add a predict button
        predict_button = Button(root, text="Predict", command=self.predict_digit)
        predict_button.pack(side='right', padx=10)

        # Add a label to show the prediction result
        self.result_label = Label(root, text="Draw a digit and press Predict")
        self.result_label.pack(pady=10)

        self.canvas.bind('<B1-Motion>', self.paint)

        # Initialize an image to draw on
        self.image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+8, y+8, fill='black')
        self.draw.ellipse([x, y, x+8, y+8], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, 280, 280], fill='white')

    def predict_digit(self):
        # Convert the image to 28x28 for the model
        image = self.image.crop((0, 0, 280, 280))
        image = image.resize((28, 28))
        image = ImageOps.invert(image)  # Invert the image

        # Convert to numpy array and preprocess
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions

        # Predict the digit
        predictions = model.predict(image, verbose=1)
        predicted_digit = np.argmax(predictions)

        # Update the result label
        self.result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Create the main window and run the application
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
