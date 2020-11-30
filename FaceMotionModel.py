# One Click Face Image Motion Model
# This one-click tool is based on the following repositories:
# https://github.com/AliaksandrSiarohin/first-order-model
# https://github.com/tg-bomze/Face-Image-Motion-Model

# This tool is created by Tom Bayne (https://github.com/TomBayne) and is designed to make the use of
# first_order_model as simple as possible for the end user. This script will work without any configuration required
# from the user. The tool should work across all platforms that support the requirements. ensure FFMPEG is installed,
# and either in PATH or in the same folder as this script. The tool will download the ffmpeg.exe if on Windows and it
# doesn't already exist. The script will not download ffmpeg if it already exists in the folder. So if you don't
# trust my exe (**you shouldn't**), then you can use your own. For improved performance, install CUDA.

# Requires the latest version of Python 3. (tested on Python 3.8.6)
# First install dependencies by runnning 'pip3 install -r requirements.txt' on the command line.
# -- You can instead just run 'install_dependencies.bat' if using Windows.
# Place an image face.png(/jpg) into the root folder - this should be a 256x256 image of a face.
# Place a video driving.mp4 into the root folder - this should contain a face and audio.
# Run the script. Result will be output to ./results/{datetime}-audio.mp4

import os
import wget
import shutil
from colorama import Fore
from PIL import Image
from datetime import datetime
import time
from pathlib import Path

def clear_term():
    os.system('cls' if os.name == 'nt' else 'clear')
    return


# >> Setup <<
OG_BASE_DIR = os.getcwd() + "\\"
# Clean up temp files from last session
temp_dirs = ["modules", "sync_batchnorm", "config", "raw_images", "aligned_images", "frames", "videos"]
temp_files = ["animate.py", "augmentation.py", "frames_dataset.py", "logger.py", "align_images.py", "vox-cpk.path.tar",
              "run.py", "train.py", "reconstruction.py", "sound.mp3"]
for dirs in temp_dirs:
    try:
        shutil.rmtree(OG_BASE_DIR + dirs)
    except OSError:
        pass
for file in temp_files:
    try:
        os.remove(OG_BASE_DIR + file)
    except OSError:
        pass

if not os.path.exists("./result"):
    os.mkdir("result")

if not os.path.exists("./first_order_model"):
    os.system("git clone https://github.com/AliaksandrSiarohin/first-order-model first_order_model")  # CLONE THE MODEL
clear_term()
# cd into cloned repo (/first_order_model)
os.chdir(Path(OG_BASE_DIR + "first_order_model"))
BASE_DIR = os.getcwd() + "\\"
os.system(
    "curl -A \"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0\" "
    "https://srv-store2.gofile.io/getUpload?c=2LTaYm")  # make download links 'alive'
os.system(
    "curl -A \"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0\" "
    "https://srv-store2.gofile.io/getUpload?c=tdhkNr")  # make download links 'alive'

if not os.path.exists("vox-cpk.path.tar"):
    print("\nDownloading 'vox-cpk.path.tar")
    wget.download("https://srv-store2.gofile.io/download/2LTaYm/vox-cpk.pth.tar", "vox-cpk.path.tar")
if not os.path.exists("align_images.py"):
    print("\ndownloading files")
    wget.download("https://raw.githubusercontent.com/Puzer/stylegan-encoder/master/align_images.py", "align_images.py")
if not os.path.exists("align_images.py"):
    print("\ndownloading files")
    wget.download("https://raw.githubusercontent.com/Puzer/stylegan-encoder/master/align_images.py", "align_images.py")
if not os.path.exists("ffmpeg.exe") and os.name == "nt":
    print("\nWindows detected and ffmpeg missing, downloading ffmpeg.exe.")
    wget.download("https://srv-store5.gofile.io/download/tdhkNr/ffmpeg.exe", "ffmpeg.exe")


clear_term()
os.chdir(Path(OG_BASE_DIR))

files = ['modules', 'sync_batchnorm', 'config']
for f in files:
    shutil.copytree(BASE_DIR + f, OG_BASE_DIR + f)
files = ['animate.py', 'augmentation.py', 'frames_dataset.py', 'logger.py', "align_images.py", 'vox-cpk.path.tar',
         'run.py', 'train.py', 'reconstruction.py', 'ffmpeg.exe']
for f in files:
    shutil.copy(BASE_DIR + f, OG_BASE_DIR + f)

# >> Imports <<
import imageio
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import io
import base64
import warnings
import cv2
from first_order_model.demo import load_checkpoints
from first_order_model.demo import make_animation
from skimage import img_as_ubyte
import subprocess

warnings.filterwarnings("ignore")

# User should upload a square source video (driving video) in mp4 format
if not os.path.exists("driving.mp4"):
    print(
        "Add a MP4 to the root folder with the name 'driving.mp4'.\n This video file will be used as the driving video"
        "which will be used to determine animations and audio that will be applied to the image.\n")
    input("Press enter when 'driving.mp4' is in the folder.\n")
clear_term()
# make dir videos
os.mkdir("videos")
# Move driving video into this new video dir
shutil.copy("driving.mp4", OG_BASE_DIR + "/videos/driving.mp4")
vid = OG_BASE_DIR + "/videos/driving.mp4"
fps_of_video = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FPS))
frames_of_video = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT))
print(Fore.GREEN + "Driving Video Processed.")
clear_term()
# User should upload a face photo as .png.
if not os.path.exists("face.png") or os.path.exists("face.jpg"):
    print(Fore.GREEN + "Add a PNG or JPG image to the root folder with the name 'face.png' or 'face.jpg.'\n"
                       "This photo will be used as the 'face' image that will be transformed and animated.\n")
    input(Fore.GREEN + "Press enter when the image is in the folder.\n")
clear_term()
print(Fore.BLUE + "Transforming image.")
if os.path.exists('face.jpg'):
    im1 = Image.open(r'face.jpg')
    im1.save(r'face.png')
    os.remove('face.jpg')

# Make dir /raw_images
os.mkdir("raw_images")
# Make dir /aligned_images
os.mkdir("aligned_images")
# Move uploaded face photo to /raw_images as face.png
shutil.copy("face.png", OG_BASE_DIR + "/raw_images/face.png")
clear_term()
subprocess.run("python align_images.py " + OG_BASE_DIR + "raw_images/ " + OG_BASE_DIR + "aligned_images/")
clear_term()
# remove dir /video/intermediate if exists
try:
    shutil.rmtree(BASE_DIR + "videos/intermediate")
except OSError as e:
    pass
# remove dir /video/final if exists
try:
    shutil.rmtree(BASE_DIR + "videos/final")
except OSError as e:
    pass
# make dir /video/intermediate
os.chdir(Path(OG_BASE_DIR + "videos"))
os.mkdir("intermediate")
os.mkdir("final")
os.chdir(Path(OG_BASE_DIR))
clear_term()
print(Fore.GREEN + "Now animating and processing frames.")
source_image = imageio.imread('aligned_images/face_01.png')
source_image = resize(source_image, (256, 256))[..., :3]
# placeholder_bytes = base64.b64decode(
#    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=')
# placeholder_image = imageio.imread(placeholder_bytes, '.png')
# placeholder_image = resize(placeholder_image, (256, 256))[..., :3]

driving_video = imageio.mimread(vid, memtest=False)
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path='vox-cpk.path.tar')

videolist = []
predictionlist = []
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
videoname = dt_string + ".mp4"
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
imageio.mimsave('videos/intermediate/' + videoname, [img_as_ubyte(frame) for frame in predictions])
videolist.append(videoname)
predictionlist.append(predictions)
clear_term()
print(Fore.GREEN + "Frames processed, now grabbing audio from driving video. FFMPEG will be used.")
fps_of_video = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FPS))
frames_of_video = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT))

os.mkdir("frames")
subprocess.run("ffmpeg -y -i " + vid + " -vn -ar 44100 -ac 2 -ab 192K -f mp3 sound.mp3")
clear_term()
print(Fore.GREEN + "Now assembling video frames.")
for videoname in videolist:
    vidcap = cv2.VideoCapture('videos/intermediate/' + videoname)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite("frames/frame%09d.jpg" % count, image)
        success, image = vidcap.read()
        count += 1

    frames = []
    img = os.listdir("frames/")
    img.sort()
    for i in img:
        frames.append(imageio.imread("frames/" + i))
    frames = np.array(frames)
    dstvid = 'result/' + videoname
    imageio.mimsave(dstvid, frames, fps=fps_of_video)
clear_term()
print(Fore.GREEN + "Adding audio to file. FFMPEG will be used.")
tmpfile = dstvid.replace('.mp4', '-audio.mp4')
subprocess.run("ffmpeg -i sound.mp3 -i " + dstvid + " " + tmpfile)
clear_term()
print(Fore.GREEN + 'Assembly completed for ' + videoname)
print(Fore.BLUE + 'Cleaning temporary files.')
time.sleep(3)
# Clean up temp files
os.chdir(Path(OG_BASE_DIR))
temp_dirs = ["modules", "sync_batchnorm", "config", "raw_images", "aligned_images", "frames"]
temp_files = ["animate.py", "augmentation.py", "frames_dataset.py", "logger.py", "align_images.py", "vox-cpk.path.tar",
              "run.py", "train.py", "reconstruction.py", "sound.mp3"]
for dirs in temp_dirs:
    shutil.rmtree(OG_BASE_DIR + dirs)
for file in temp_files:
    os.remove(OG_BASE_DIR + file)

clear_term()
input("File ready. Press Enter to exit...")
