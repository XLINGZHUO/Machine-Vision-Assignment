#!/usr/bin/env python3
 
from genericpath import exists
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pycocotools import mask
from PIL import Image
import codecs
from glob import glob
import shutil
 
part = "train"   # train  test

ROOT_DIR = 'C:/Users/awei/Desktop/detection'
IMAGE_DIR = os.path.join(ROOT_DIR, "Image")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "GT")

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files
 
 
def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_.*'   # 用于匹配对应的二值mask
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


saved_path = "VOC2007/"
# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

ftrainval = open('VOC2007/ImageSets/Main/trainval.txt', 'a')   # 'a'为append
ftest = open('VOC2007/ImageSets/Main/test.txt', 'w')  
ftrain = open('VOC2007/ImageSets/Main/train.txt', 'w')  
fval = open('VOC2007/ImageSets/Main/val.txt', 'w')

def splitData():
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            if not os.path.exists(saved_path+ "JPEGImages/"+os.path.basename(image_filename)):
                shutil.copy(image_filename, saved_path + "JPEGImages/")
            name = os.path.basename(image_filename).split('.')[0]
            if not os.path.exists(saved_path + "Annotations/"+name+".xml"):
                print("not exist:"+name+".xml")
                continue
            if part=="train":
                ftrain.write(name)
                ftrain.write('\n')
            else:
                fval.write(name)
                fval.write('\n')
            ftrainval.write(name)
            ftrainval.write('\n')

    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()

def main():
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            width,height,channels = image.size[0],image.size[1],image.layers


            boxList = []
            labels=[]
            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
 
                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)

                    name = os.path.basename(annotation_filename)
                    image_name = name.split('_')[-3]
                    label = name.split('_')[1] # 

                    
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    if image is not None:
                        binary_mask = pycococreatortools.resize_binary_mask(binary_mask, image.size)
                    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

                    bounding_box = mask.toBbox(binary_mask_encoded)
                    boxList.append([bounding_box[0],bounding_box[1],bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]])
                    labels.append(label)

            # 读取标注信息并写入 xml
            with codecs.open(saved_path + "Annotations/" + image_name + ".xml", "w", "utf-8") as xml:
        
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'WH_data' + '</folder>\n')
                xml.write('\t<filename>' + image_name + ".jpg" + '</filename>\n')
                xml.write('\t<source>\n')
                xml.write('\t\t<database>WH Data</database>\n')
                xml.write('\t\t<annotation>WH</annotation>\n')
                xml.write('\t\t<image>flickr</image>\n')
                xml.write('\t\t<flickrid>NULL</flickrid>\n')
                xml.write('\t</source>\n')
                xml.write('\t<owner>\n')
                xml.write('\t\t<flickrid>NULL</flickrid>\n')
                xml.write('\t\t<name>WH</name>\n')
                xml.write('\t</owner>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
                for box,label in zip(boxList,labels):

                    xmin, ymin, xmax,ymax = box

                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>' + label+ '</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                        print(image_filename, xmin, ymin, xmax, ymax, label)
                xml.write('</annotation>')

if __name__ == "__main__":
    main()  # 用于将mask生成xml
    splitData() # 用于切分数据（适用非随机的切分）
