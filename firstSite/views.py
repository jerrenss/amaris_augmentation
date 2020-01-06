from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
#Additional imports
import os
import zipfile
import matplotlib
matplotlib.use('agg')
from matplotlib.image import imread
from PIL import Image
import sys
import logging
from firstSite.pyfiles import augmentation as aug
import uuid
import shutil
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def homepage(request):
    # return HttpResponse('AMARIS AI')
    return render(request, 'homepage.html')


'''
APPLICATION DETAILS:

REDIRECTS:
Homepage <-> Single Folder Augmentation
Homepage <-> Dual Folder Augmentation 

SINGLE FOLDER AUGMENTATION:
1. Generate a unique session key, store as variable 'ID'
2. Choose files to upload
3. Upon button click:
    a. Upload all files from a client-side selected folder to Path = (MEDIA_ROOT + ID + '/Original')
    b. Store Path + options chosen in a dictionary.
4. Peform augmentation using Start function, passing in a dictionary. 
5. A list of 3D numpy arrays are returned
6. Save them into Path = (MEDIA_ROOT + ID + '/Augmented')



'''
def singleA(request):
    modes = ['Random', 'Manual']
    types = ['Rotate','Blur', 'Colour-Grade', 'Noise', 'Shear', 'Scale', 'Flip-Horizontal','Flip-Vertical']
    if request.method == "POST":
        mode = request.POST.getlist('step1')
        manualTypes = request.POST.getlist('step2')
        uploaded_file = request.FILES.getlist('uploadfile')

        session_key = str(uuid.uuid1())
        storagePath = os.path.join(settings.MEDIA_ROOT, session_key, 'original')
        storagePath2 = os.path.join(settings.MEDIA_ROOT, session_key, 'augmented')
        os.makedirs(storagePath2)

        fs = FileSystemStorage(location = storagePath)
        
        for files in uploaded_file:
            fs.save(files.name, files)

        # Get file names in directory
        names = sorted(os.listdir(storagePath))
        names_final = []
        supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff']
        for name in names:
            parts = name.split('.')
            if parts[1].lower() in supported_formats:
                names_final.append(name)

        dict1 = {'folder': [storagePath], 'mode': mode, 'options' : manualTypes}
        try:
            pass
            numpy4d = start(dict1)
        except Exception as e:
            # print(str(e))
            return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag2': True})
        save(numpy4d, storagePath2, names_final)
        # Zip Folder for download
        shutil.make_archive('Augmented', "zip", os.path.join(settings.MEDIA_ROOT, session_key, 'augmented'))
        path_to_file = BASE + '/Augmented.zip' 
        response = HttpResponse(open(path_to_file, 'rb'), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=Augmented.zip'
        # Remove session folder
        shutil.rmtree(os.path.join(settings.MEDIA_ROOT, session_key), ignore_errors=True)
        return response
        # return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag3': True})
    else:
        return render(request, 'single_aug.html', {'types': types, 'modes': modes})

def save(numpy4d, filepath, namelist):
    # counter = 0
    for i in range(len(numpy4d)):
        # name = str(counter)
        temp = Image.fromarray(numpy4d[i], 'RGB')
        temp.save(filepath + '\\augmented_' + namelist[i])
        # counter = counter + 1

def dualA(request):
    modes = ['Random', 'Manual']
    types = ['Rotate','Blur', 'Colour-Grade', 'Noise', 'Shear', 'Scale', 'Flip-Horizontal','Flip-Vertical']
    if request.method == "POST":
        mode = request.POST.getlist('step1')
        manualTypes = request.POST.getlist('step2')
        uploaded_file = request.FILES.getlist('uploadfile')
        uploaded_file2 = request.FILES.getlist('uploadfile2')

        session_key = str(uuid.uuid1())
        storagePath = os.path.join(settings.MEDIA_ROOT, session_key, 'original')
        storagePathA = os.path.join(settings.MEDIA_ROOT, session_key, 'augmented', 'first')
        storagePath2 = os.path.join(settings.MEDIA_ROOT, session_key, 'original2')
        storagePath2A = os.path.join(settings.MEDIA_ROOT, session_key, 'augmented', 'second')
        os.makedirs(storagePathA)
        os.makedirs(storagePath2A)
       
        fs = FileSystemStorage(location = storagePath)
        fs2 = FileSystemStorage(location = storagePath2)

        for files in uploaded_file:
            print(type(files))
            fs.save(files.name, files)

        for files in uploaded_file2:
            fs2.save(files.name, files)

        # Get file names in directory
        names = sorted(os.listdir(storagePath))
        names_final = []
        supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff']
        for name in names:
            parts = name.split('.')
            if parts[1].lower() in supported_formats:
                names_final.append(name)

        names2 = sorted(os.listdir(storagePath2))
        names_final2 = []
        for name in names2:
            parts = name.split('.')
            if parts[1].lower() in supported_formats:
                names_final2.append(name)
        

        dict1 = {'folder': [storagePath, storagePath2], 'mode': mode, 'options' : manualTypes}
        try:
            pass
            numpy4d, numpy4d2 = start(dict1)
        except Exception as e:
            # print(str(e))
            return render(request, 'single_aug.html', {'types': types, 'modes': modes, 'flag2': True})
        save(numpy4d, storagePathA, names_final)
        save(numpy4d2, storagePath2A, names_final2)
        # Zip Folder for download
        shutil.make_archive('Augmented_Dual', "zip", os.path.join(settings.MEDIA_ROOT, session_key, 'augmented'))
        path_to_file = BASE + '/Augmented_Dual.zip' 
        response = HttpResponse(open(path_to_file, 'rb'), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=Augmented_Dual.zip'
        # Remove session folder
        shutil.rmtree(os.path.join(settings.MEDIA_ROOT, session_key), ignore_errors=True)
        return response
        return render(request, 'dual_aug.html', {'types': types, 'modes': modes, 'flag3': True})
    else:
        return render(request, 'dual_aug.html', {'types': types, 'modes': modes})


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