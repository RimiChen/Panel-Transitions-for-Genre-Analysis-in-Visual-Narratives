from PIL import Image
import sys 

path = sys.argv[1] 
print(sys.argv[1])
image = Image.open(path)
image.show()