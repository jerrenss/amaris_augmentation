import easygui
import os
import zipfile
from matplotlib.image import imread
from PIL import Image


# Function for opening files from a zipfile input
def read_zip_file():
    filepath = easygui.fileopenbox()
    zfile = zipfile.ZipFile(filepath)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        image = imread(ifile)
        print(image.shape)

# Function for opening files from a folder directory input
def read_from_folder():
    path = easygui.diropenbox()
    print(path)
    list1 = os.listdir(path)
    # for name in list1:
    #     image = imread(path + '/' + name)
    #     print(image.shape)

# Function for opening a file
def read_file():
    filepath = (easygui.fileopenbox())
    image = imread(filepath)
    print(image.shape)

#Function for displaying a file
def open_file():
    filepath = (easygui.fileopenbox())
    os.system(filepath)

# Function to select a folder directory to save files into
def save_to_folder(numpy4d):
    filepath = easygui.diropenbox()
    for i in range(len(numpy4d)):
        temp = Image.fromarray(numpy4d[i-1], 'RGB')
        temp.save(filepath + 'tempid.png', 'PNG')

# Function to concatenate a suffix to a string
def addSuffix(current):
    current = current + '_augmented'


def main(dictionary):
    pass

read_from_folder()
