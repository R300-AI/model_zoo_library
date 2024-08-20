import os

def import_or_install(package):
  try:
      import cv2
  except ImportError:
      os.system('pip install {package}')
