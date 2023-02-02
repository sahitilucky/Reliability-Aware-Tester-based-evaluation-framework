import os
import glob

ENGINE_LIST = [os.path.basename(f)[:-3]
               for f in glob.glob(os.path.dirname(__file__)+"/*.py")
               if '__init__' not in f]
