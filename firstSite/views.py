from django.http import HttpResponse
from django.shortcuts import render
#Additional imports
from subprocess import run, PIPE
from django import forms
import os
import easygui
import zipfile
from matplotlib.image import imread
from PIL import Image
import sys
import logging
from firstSite.pyfiles import augmentation as aug
import tkinter
from tkinter import filedialog, messagebox


def homepage(request):
    # return HttpResponse('AMARIS AI')
    return render(request, 'homepage.html')


def singleA(request):
    modes = ['Random', 'Manual']
    types = ['Rotate','Blur', 'Colour-Grade', 'Noise', 'Shear', 'Scale', 'Flip-Horizontal','Flip-Vertical']
    if request.method == "POST":
        mode = request.POST.getlist('step1')
        manualTypes = request.POST.getlist('step2')
        location = read()
        print(location)
        if location == '':
            # showError("Error. Please Select Folder Again.")
            return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag1': True})
        name_list = os.listdir(location)
        name_list.sort()
        dict1 = {'folder': [location], 'mode': mode, 'options' : manualTypes}
        try:
            numpy4d = start(dict1)
        except Exception as e:
            # showError((str(e)))
            return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag2': True})
        save(numpy4d, name_list)
        return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag3': True})
    else:
        return render(request, 'single_aug.html', {'types': types, 'modes': modes})


def dualA(request):
    modes = ['Random', 'Manual']
    types = ['Rotate','Blur', 'Colour-Grade', 'Noise', 'Shear', 'Scale', 'Flip-Horizontal','Flip-Vertical']
    if request.method == "POST":
        mode = request.POST.getlist('step1')
        manualTypes = request.POST.getlist('step2')
        location = read()
        location2 = readMask()
        if location == '' or location2 == '':
            # showError()
            return render(request, 'dual_aug.html', {'types': types, 'modes': modes, 'flag0': True})
        name_list = os.listdir(location)
        name_list.sort()
        name_list2 = os.listdir(location2)
        name_list2.sort()        
        if name_list != name_list2:
            return render(request, 'dual_aug.html', {'types': types, 'modes': modes, 'flag1': True})
        dict1 = {'folder': [location, location2], 'mode': mode, 'options' : manualTypes}
        try:
            numpy4d, numpy4d_m = start(dict1)
        except Exception as e:
            return render(request, 'dual_aug.html', {'types': types, 'modes': modes, 'flag2': True})
        save(numpy4d, name_list)
        save_m(numpy4d_m, name_list)
        # showSuccess()
        return render(request, 'dual_aug.html', {'types': types, 'modes': modes, 'flag3': True})
    else:
        return render(request, 'dual_aug.html', {'types': types, 'modes': modes})


def start(dict):
    folder_list = dict["folder"]
    mode_list = dict["mode"]
    options_list = dict["options"]
    if len(folder_list) == 1:
        # single
        folder_path = folder_list[0]
        # check mode random or manual
        if mode_list[0] == "Random":
            #do single random
            return aug.rand_aug_single(folder_path)
        else:
            # do single manual
            return aug.manual_augment_single(folder_path, options_list)
    else:
        # multiple
        pass

def read_from_folder():
    path = easygui.diropenbox()
    return path

def save(numpy4d, name_list):
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    currdir = os.getcwd()
    filepath = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please Select SAVING Directory for IMAGE')
    counter = 0
    for i in range(len(numpy4d)):
        name = name_list[i]
        temp = Image.fromarray(numpy4d[i], 'RGB')
        temp.save(filepath + '\\' + name, 'PNG')
        counter = counter + 1

def save_m(numpy4d, name_list):
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    currdir = os.getcwd()
    filepath = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please Select SAVING Directory for MASKS')
    counter = 0
    for i in range(len(numpy4d)):
        name = name_list[i]
        temp = Image.fromarray(numpy4d[i], 'RGB')
        temp.save(filepath + '\\' + name, 'PNG')
        counter = counter + 1


def read():
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please Select IMAGE Directory')
    return tempdir

def readMask():
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please Select MASK Directory')
    return tempdir

def showSuccess():
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    messagebox.showinfo("Augmentation", "Succesfully Augmented! :)")

def showError(message):
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)
    messagebox.showinfo("Augmentation", message)

def start(dict):
    """dict should contain only the fields in the exact format:
        {"folder":[...],
         "mode":[<Either "Random" or "Manual"],
         "options":[<Any one of: 
            "Flip Horizontal"
            "Flip Vertical"
            "Blur"
            "Contrast"
            "Noise"
            "Color"
            "Rotate"
            "Shear"
            "Scale"
            >] }
    """
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    folder_list = dict["folder"]

    mode_list = dict["mode"]

    options_list = dict["options"]

    augmented_images = []
    if len(folder_list) == 1:
        # Go to the next stage
        # If the length of the folder list is 1, means that there is only a single folder to process
        print("Start processing for single batch")
        augmented_images = process_single(folder_list[0], mode_list, options_list)
        return augmented_images
    else: 
        print("Start processing for multiple batch")
        augmented_images_1, augmented_images_2 = process_multiple(folder_list, mode_list, options_list)
        return augmented_images_1, augmented_images_2

# Processes a single batch of images
def process_single(folder_path, mode_list, options_list):
    # Next stage, check if random or manual
    mode = mode_list[0]
    # check if random or manual
    if mode == "Random":
        print("processing single batch (RANDOM)")
        return aug.single_random(folder_path)
    else:
        # Mode is manual
        print("processing single batch (MANUAL)")
        return aug.single_manual(folder_path, options_list)

# Processes a multiple (2) batches of images
def process_multiple(folder_list, mode_list, options_list):
    # check if its random or manual
    mode = mode_list[0]
    if mode == "Random":
        print("processing multiple batch (RANDOM)")
        #return aug.multiple_random(folder_list)
        return aug.multiple_random_mask(folder_list)
    else: 
        print("processing multiple batch (MANUAL)")
        #return aug.multiple_manual(folder_list, options_list)
        return aug.multiple_manual_mask(folder_list, options_list)