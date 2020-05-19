import numpy as np
import cv2
import sys
import os

def load_data(data_path,new_path):
    new_rows = 24
    new_cols = 24
    file_ext = ".pgm"
    count = 0
    folders = os.listdir(data_path)
    for fldr in folders:
        files = os.listdir(data_path+fldr)

        for fl in files:
            if (fl[-4:] != file_ext):
                continue

            full_path = data_path+fldr+"/"+fl

            img = cv2.imread(full_path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(color,(new_cols,new_rows))
            mirror = cv2.flip(resized,1)

            cv2.imwrite(new_path+"img_" + str(count).zfill(6)+".ppm",resized)
            count += 1
            cv2.imwrite(new_path+"img_" + str(count).zfill(6)+".ppm",mirror)
            count += 1
        
def main():
    if (len(sys.argv) != 3):
        print("need current path and new path")
        sys.exit()

    data_path = sys.argv[1]
    new_path = sys.argv[2]
        
    load_data(data_path,new_path)

if __name__ == "__main__":
    main()
