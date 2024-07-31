import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_loss():
    try:
        input_values = [float(entries[0].get()), int(entries[1].get()), float(entries[2].get()), float(entries[3].get()), float(entries[4].get())]
        input_data = torch.tensor([input_values], dtype=torch.float32)
        with torch.no_grad():
            prediction = net(input_data)
        result_label.config(text=f"Rac/Rdc: {prediction.item()}")
    except ValueError:
        result_label.config(text="Please enter valid input values")


def plot_predictions():
    try:
        # get input
        freq = float(entries[0].get())
        nlayer = int(entries[1].get())
        lg = float(entries[2].get())
        kdx_values = [0, 0.5, 1]
        hma_values = np.linspace(0, 1, 100)

        for kdx in kdx_values:
            predictions = []
            for hma in hma_values:
                input_data = torch.tensor([[freq, nlayer, lg, kdx, hma]], dtype=torch.float32)
                with torch.no_grad():
                    prediction = net(input_data)
                predictions.append(prediction.item())
            ax.plot(hma_values, predictions, label=f'kdx = {kdx}')
        ax.set_xlabel('hma/hmcl')
        ax.set_ylabel('Rac/Rdc')
        ax.set_title('Prediction Curve')
        ax.legend()
        canvas.draw()
    except Exception as e:
        print(f"Error: {e}")
def plot_frequency_response():
    try:
        # get input
        freq_values = np.linspace(0.1, 1, 100)
        nlayer = int(entries[1].get())
        lg = float(entries[2].get())
        kdx = float(entries[3].get())
        hma = float(entries[4].get())

        rac_rdc_values = []
        for freq in freq_values:
            input_data = torch.tensor([[freq, nlayer, lg, kdx, hma]], dtype=torch.float32)
            with torch.no_grad():
                prediction = net(input_data)
            rac_rdc_values.append(prediction.item())

        ax_freq.clear()
        ax_freq.plot(freq_values, rac_rdc_values, label=f'nlayer: {nlayer}, lg: {lg}, kdx: {kdx}, hma: {hma}')
        ax_freq.set_xlabel('Frequency [MHz]')
        ax_freq.set_ylabel('Rac/Rdc')
        ax_freq.set_title('Frequency Response of Rac/Rdc')
        ax_freq.legend()
        canvas_freq.draw()
    except Exception as e:
        print(f"Error: {e}")

# load NN
class Net(nn.Module):
    def __init__(self, num_features, num_hidden1, num_hidden2, num_hidden3, num_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.fc4 = nn.Linear(num_hidden3, num_output)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

net = torch.load('F_n6.pth')
net.eval()

# main
root = tk.Tk()
root.title("AC resistance Calculator")
root.geometry("1024x1000")

# frame
left_upper_frame = tk.Frame(root, padx=5, pady=5)
left_lower_frame = tk.Frame(root, padx=5, pady=5)
right_upper_frame = tk.Frame(root, padx=5, pady=5)
right_lower_frame = tk.Frame(root, padx=5, pady=5)

left_upper_frame.grid(row=0, column=0, sticky="nsew")
left_lower_frame.grid(row=1, column=0, sticky="nsew")
right_upper_frame.grid(row=0, column=1, sticky="nsew")
right_lower_frame.grid(row=1, column=1, sticky="nsew")

# frame weight
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# inner weight in 4 window
left_upper_frame.rowconfigure(0, weight=1)
left_upper_frame.columnconfigure(0, weight=1)
left_lower_frame.rowconfigure(0, weight=1)
left_lower_frame.columnconfigure(0, weight=1)
right_upper_frame.rowconfigure(0, weight=1)
right_upper_frame.columnconfigure(0, weight=1)
right_lower_frame.rowconfigure(0, weight=1)
right_lower_frame.columnconfigure(0, weight=1)

#Labelframe
input_info = ttk.Labelframe(left_upper_frame, text='Design inputs', padding=10)
input_info.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

labels = ["Freq [MHz]", "nlayer", "lg [mm]", "kdx", "hma/hmcl"]
entries = []
for i, label in enumerate(labels):
    tk.Label(input_info, text=label).grid(row=i, column=0, padx=10, pady=10)
    entry = tk.Entry(input_info)
    entry.grid(row=i, column=1, padx=10, pady=10)
    entries.append(entry)

input_info.columnconfigure(1, weight=1)

result_label = tk.Label(input_info, text="Rac/Rdc: Waiting for input...")
result_label.grid(row=6, column=0, columnspan=2)




calculate_button = ttk.Button(input_info, text="Calculate", command=calculate_loss)
calculate_button.grid(row=7, column=0, columnspan=2)

# figure
image_frame = ttk.Labelframe(left_lower_frame, text='Geometry', padding=10)
image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
img = tk.PhotoImage(file='geom2.png')
img_label = tk.Label(image_frame, image=img)
img_label.pack(fill='both', expand=True)

# plot1
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=right_upper_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, sticky="nsew")

# plot2
fig_freq, ax_freq = plt.subplots()
canvas_freq = FigureCanvasTkAgg(fig_freq, master=right_lower_frame)
canvas_freq_widget = canvas_freq.get_tk_widget()
canvas_freq_widget.grid(row=0, column=0, sticky="nsew")


plot_button = ttk.Button(right_upper_frame, text="Plot Predictions", command=plot_predictions)
plot_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)


plot_freq_button = ttk.Button(right_lower_frame, text="Plot Frequency Response", command=plot_frequency_response)
plot_freq_button.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

root.mainloop()
