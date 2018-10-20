from tkinter import filedialog
from tkinter import *

def BVideoFunction():
    global videoVar
    videoVar =  filedialog.askopenfilename(initialdir = "/",title = "Choose Video source",filetypes = (("avi files","*.avi"),("mp4 files","*.mp4"),("all files","*.*")))

def BOutpDirFunction():
    global outputDir
    outputDir =  filedialog.askdirectory()

def BRunFunction():
    global videoVar
    global outputDir
    root.destroy()

root = Tk(className = ' Face Classification From Video')
root.configure(background='#A3CEDC')

BVideo = Button(root, text ="    Choose Video source    ", command = BVideoFunction)
BVideo.grid(ipadx=3, ipady=3, padx=4, pady=4)

BOutpDir = Button(root, text ="Choose Output directory \n(Needs to be empty)", command = BOutpDirFunction)
BOutpDir.grid(ipadx=2, ipady=2, padx=4, pady=4)

LFrames = Label( root, text='Frames after which\n to exrtact image:' )
LFrames.grid(column=2, row=0, ipadx=2, ipady=2, padx=4, pady=4)
frames = StringVar()
frames.set(50)
EFrames = Entry(root, bd =5, textvariable = frames)
EFrames.grid(column=3, row=0, ipadx=2, ipady=2, padx=4, pady=4)

LConstant = Label( root, text='Silhouette constant for locating outliers\n(Recommendation: Do not modify):' )
LConstant.grid(column=2, row=1, ipadx=1, ipady=1, padx=4, pady=4)
constant = StringVar()
constant.set(0.1)
EConstant = Entry(root, bd =5, textvariable = constant)
EConstant.grid(column=3, row=1, ipadx=2, ipady=2, padx=4, pady=4)

BRun = Button(root, text ="RUN", command = BRunFunction)
BRun.grid(column=2, row=3, ipadx=3, ipady=3, padx=4, pady=4)

try:#sto run
    print (videoVar)
    print (outputDir)
except NameError: 
    print("vale label")

root.mainloop()
print (frames.get())