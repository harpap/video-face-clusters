from tkinter import *

root = Tk()
frame=Frame(root)
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)
frame.grid(row=0, column=0, sticky=N+S+E+W)
grid=Frame(frame)
grid.grid(sticky=N+S+E+W, column=0, row=7, columnspan=2)
Grid.rowconfigure(frame, 7, weight=1)
Grid.columnconfigure(frame, 0, weight=1)

#example values
for x in range(10):
    for y in range(5):
        btn = Button(frame)
        btn.grid(column=x, row=y, sticky=N+S+E+W)

for x in range(10):
  Grid.columnconfigure(frame, x, weight=1)

for y in range(5):
  Grid.rowconfigure(frame, y, weight=1)

root.mainloop()
'''
if not dispose == -1:
        best_cl -= 1
        os.rmdir(output_summary[dispose])
        del output_summary[dispose]
        outl_dir = os.path.expanduser(output_dir + '/outliers')
        if not os.path.exists(outl_dir):
            os.makedirs(outl_dir)
        move(output_dir_cluster[dispose],outl_dir)
        move(output_dir_cluster[dispose]+' (cropped)',outl_dir)
        i=0
        while i < nrof_images:
            if data_list[i].cluster_label == dispose:
                del data_list[i]
                nrof_images-=1
                i-=1
            i+=1
            
'''