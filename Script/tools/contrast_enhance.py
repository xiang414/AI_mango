# coding: utf-8

from PIL import Image
from PIL import ImageEnhance
 
#原始圖像
image = Image.open('D:\\AI_Mango\\data\\Costom Data\\train\\00456.jpg')
image = image.resize((264, 264), Image.ANTIALIAS)
#image.show()
image = image.crop([20,20,244,244])
image.show()
 
#亮度增强
enh_bri = ImageEnhance.Brightness(image)
brightness = 2
image_brightened = enh_bri.enhance(brightness)
#image_brightened.show()
 
#色度增强
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored = enh_col.enhance(color)
image_colored.show()
 
#對比度增强
enh_con = ImageEnhance.Contrast(image_colored)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()
 
#銳度增强
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 3.0
image_sharped = enh_sha.enhance(sharpness)
#image_sharped.show()
