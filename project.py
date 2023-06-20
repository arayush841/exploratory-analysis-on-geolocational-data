from tkinter import *
import tkinter as tk
from tkinter import ttk
import webview

app = tk.Tk()
app.geometry("750x250")

app.title("Exploratory Data Analysis on Geolocation Data")

bg = tk.PhotoImage(file="Images/bg.png")
myLabel = Label(app, image=bg)
myLabel.place(x=0, y=0)


def open_browser(*args):

    # webbrowser.open_new("http://127.0.0.1:5500/Cities/Bangalore.html")
    webview.create_window(
        '{}'.format(cb1.get()), 'http://127.0.0.1:5500/Cities/{}.html'.format(cities[int(cb1.current())]))
    webview.start()


cities = ("Bangalore", "Chennai", "Delhi",
          "Gurgaon", "Hyderabad", "Kolkata", "Mumbai")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
cb1 = ttk.Combobox(app, values=cities, width=10, font="Verdana 16")
# cb1.grid(row=1, column=0, padx=10, pady=10)
cb1.place(x=280, y=100)

b1 = tk.Button(app, text='Go', command=lambda: open_browser(), font="16")
b1.place(x=445,  y=100)
cb1.set("Select City")
app.mainloop()
