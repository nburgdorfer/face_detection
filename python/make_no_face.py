import numpy as np
import cv2
import sys
import os

def load_data(data_path,new_path):
    new_rows = 24
    new_cols = 24
    file_ext = ".jpg"
    count = 0
    files = os.listdir(data_path)
    for fl in files:
        if (fl[-4:] != file_ext):
            continue

        full_path = data_path+fl

        img = cv2.imread(full_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(color,(new_cols*4,new_rows*4))
        rows,cols,_ = resized.shape

        max_rows = int(np.floor(rows/new_rows))
        max_cols = int(np.floor(cols/new_cols))

        r=0
        while(r+new_rows<rows):
            c=0
            while(c+new_cols<cols):
                new_img = np.array(color[r:r+new_rows,c:c+new_cols])
                cv2.imwrite(new_path+"img_" + str(count).zfill(6)+".ppm",new_img)

                count+=1
                if (count > 50000):
                    sys.exit()
                c+=new_cols
            r+=new_rows


        
def main():
    if (len(sys.argv) != 3):
        print("need current path and new path")
        sys.exit()

    data_path = sys.argv[1]
    new_path = sys.argv[2]
        
    load_data(data_path,new_path)

if __name__ == "__main__":
    main()
