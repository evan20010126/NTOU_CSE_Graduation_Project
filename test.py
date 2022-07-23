import tkinter as tk

root = tk.Tk()


class mainpage(object):
    def __init__(self, master=None):
        self.root = master
        self.page = tk.Frame(self.root)
        self.page.pack()
        self.Button = tk.Button(self.page, text=u'跳頁', command=self.secpage)
        self.Button.grid(column=0, row=0)

    def secpage(self):
        self.page.destroy()
        secondpage(self.root)


class secondpage(object):
    def __init__(self, master=None):
        self.root = master
        self.page = tk.Frame(self.root)
        self.page.pack()
        self.Button = tk.Button(self.page, text=u'主頁', command=self.mainpage)
        self.Button.grid(column=0, row=0)

    def mainpage(self):
        self.page.destroy()
        mainpage(self.root)


mainpage(root)
root.mainloop()
