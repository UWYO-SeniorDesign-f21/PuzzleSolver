import tkinter as tk
from tkinter import filedialog


def uploadButtonFunction():
    path = filedialog.askopenfilename()
    uploadBox.delete(1.0, 'end')
    uploadBox.insert(1.0, path)


if __name__ == '__main__':
    # init window
    window = tk.Tk()
    window.title("Puzzle Solver")
    window.geometry("1000x700")
    window.resizable(False, False)

    # Organization paneling
    leftPanel = tk.PanedWindow(window, orient=tk.HORIZONTAL,
                               background='#595959', bg='#595959', height=700, width=350)
    leftPanel.pack(side=tk.LEFT, fill=tk.BOTH)

    mainPanel = tk.PanedWindow(
        window, orient=tk.HORIZONTAL, background='#808080', bg='#808080', height=700, width=650)
    mainPanel.pack(side=tk.RIGHT, fill=tk.BOTH)

    # file upload setup
    uploadLabel = tk.Label(
        leftPanel, text='Pieces Location:', background='#595959', fg='#cccccc')
    uploadLabel.place(x=5, y=5, anchor=tk.NW)

    global uploadBox
    uploadBox = tk.Text(leftPanel, height=1, width=30)
    uploadBox.place(x=100, y=5, anchor=tk.NW)

    uploadButton = tk.Button(leftPanel, text='Open',
                             command=uploadButtonFunction, height=1, width=10)
    uploadButton.place(x=265, y=35, anchor=tk.NW)

    # Run
    window.mainloop()
