import os

def varify_package_installed(package):
  try:
      __import__(package)
  except ImportError:
      os.system(f'pip install {package}')
