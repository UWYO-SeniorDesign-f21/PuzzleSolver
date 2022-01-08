import tkinter as tk
import tkinter.font as tkFont

if __name__ == '__main__':
    # init window
    window = tk.Tk()
    window.title("Puzzle Solver")
    window.geometry("1000x700")
    window.resizable(False, False)

    # Organization paneling
    leftPanel = tk.PanedWindow(window, orient=tk.HORIZONTAL,
                               background='#595959', bg='#595959', height=700, width=250)
    leftPanel.pack(side=tk.LEFT, fill=tk.BOTH)

    mainPanel = tk.PanedWindow(
        window, orient=tk.HORIZONTAL, background='#808080', bg='#808080', height=700, width=750)
    mainPanel.pack(side=tk.RIGHT, fill=tk.BOTH)

    # Run
    window.mainloop()
