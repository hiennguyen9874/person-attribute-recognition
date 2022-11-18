from PIL import Image
import os
import pandas as pd

directory = 'crop'
image_name = []
height = []
width = []

i = 0
for filename in os.listdir(directory):
    img_name = str(i) + '.jpg'
    im = Image.open('crop/' + img_name)
    print('Image - ' + str(i))
    print(im.size)
    w, h = im.size
    image_name.append(img_name)
    height.append(h)
    width.append(w)
    i+=1

print(image_name)
print(height)
print(width)

df = pd.DataFrame(
    {'image_name': image_name,
     'height': height,
     'width': width
    })

df.to_csv('image_size.csv', index=False)