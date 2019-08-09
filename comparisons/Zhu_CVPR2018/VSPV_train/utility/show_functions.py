# for displaying the images in a row
from io import BytesIO                                     
import PIL
from IPython.display import display, Image

def display_img_array(ima):
    im = PIL.Image.fromarray(ima)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

#import numpy as np
#from PIL import Image
#import matplotlib.image as  mpimg
##tgt_mask = np.asarray(Image.open('850_4_200x200.png'))
#tgt_mask = np.asarray(Image.open('446_17_200x200.png'))
#display_img_array(tgt_mask)