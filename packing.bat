call D:\test\visual_mouse\.venv\Scripts\activate
pyinstaller --noconsole main.py
copy conf.ini dist\main
mkdir dist\main\_internal\mediapipe\modules
xcopy resource\mediapipe\modules\ dist\main\_internal\mediapipe\modules\ /E /I
pause