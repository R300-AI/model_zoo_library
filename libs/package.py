import os

def import_with_install(package):
  try:
      __import__(package)
  except ImportError:
      os.system(f'pip install {package}')
