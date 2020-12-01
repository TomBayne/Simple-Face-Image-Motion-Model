# Simplified Face Image Motion Model
## One Click Face Image Motion Model

### Thanks to the following repositories and users for providing the base model for this script.
https://github.com/AliaksandrSiarohin/first-order-model </br>
https://github.com/tg-bomze/Face-Image-Motion-Model </br>

## Explanation.
I designed this tool to make the use of 'first-order-model' as simple as possible for the end user. This script will work without any configuration required from the user. 
The tool was designed with the following features in mind:
- The tool should work across as many platforms as possible without modification.
- Minimal configuration or setup should be required from the user.
- The tool should be able to generate complete results in a single click.
- The tool should run purely in Python, and not require Jupyter/IPython notebooks.

## Requirements
- Git
- Any recent version of Python 3. The code has been tested on 3.8.6 only.
- Pip requirements found in requirements.txt - **These can either be installed using the included .bat file on Windows, or manually using the command 'pip3 install -r requirements.txt'**
- For increased performance, you can install CUDA 10.1 to utilise any available nVidia GPU, however any modern CPU should be fast enough.
- 16GB RAM and a fast CPU (and GPU if using CUDA) is recommended.
- It is strongly advised to put your own ffmpeg.exe file into the root folder, otherwise a copy I have uploaded to GoFile.io will be downloaded instead.

## How to use
- Ensure requirements are fulfilled.
- Add a jpg/png file named 'face.png' or 'face.jpg' into the root folder. This will be used as the primary image for the output.
- Add a MP4 video file named 'driving.mp4' into the root folder. This will be used to determine animations and motion to be applied to the image.
- Run FaceMotionModel.py, or run 'run_windows.bat' if using Windows.
- Output can be found in ./results
