import rembg
import cv2 as cv

#file_names = ['5.jpg', '13.jpg', '14.jpg', '16.jpg', '21.jpg', '27.jpg', '39.jpg', '42.jpg', '56.jpg', '62.jpg']
file_names = ['1_2.jpg']

inDir = 'images/'
outDir = ''
#inFile = 'images/ori/31.jpg'
#outFile = 'out/31_test.png'

'''Remove set of images'''
for file in file_names:
    img = cv.imread(inDir+file)
    img_r = rembg.remove (img)
    cv.imwrite(outDir+file, img_r)