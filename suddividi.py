#Script Python che prende in input immagini JPG e file XML da una cartella e
#li suddivide casualmente nelle cartelle "train", "val" e "test" con una
#proporzione del 60% per il training, 20% per la validazione e 20% per il test:

import os
import random
import shutil

def split_dataset(source_folder, train_folder, val_folder, test_folder):

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    file_list = os.listdir(source_folder)
    random.shuffle(file_list)

    total_files = len(file_list)
    train_size = int(total_files * 0.6)
    val_size = int(total_files * 0.2)
    test_size = total_files - train_size - val_size

    train_files = file_list[:train_size]
    val_files = file_list[train_size:train_size + val_size]
    test_files = file_list[train_size + val_size:]

    move_files(train_files, source_folder, train_folder)
    move_files(val_files, source_folder, val_folder)
    move_files(test_files, source_folder, test_folder)

def move_files(file_list, source_folder, destination_folder):
    for file in file_list:
        if file.endswith('.jpg'):
            base_name = os.path.splitext(file)[0]
            xml_file = base_name + '.xml'
            if xml_file in file_list:
                source_jpg = os.path.join(source_folder, file)
                source_xml = os.path.join(source_folder, xml_file)
                dest_jpg = os.path.join(destination_folder, file)
                dest_xml = os.path.join(destination_folder, xml_file)
                shutil.move(source_jpg, dest_jpg)
                shutil.move(source_xml, dest_xml)

oggetti = ["car", "dog", "horse", "pig", "cow", "duck", "sheep", "train", 
"orange", "banana", "apple", "background"]
for oggetto in oggetti:
  source_folder = "/content/drive/MyDrive/etichettate/" + oggetto
  train_folder = "/content/drive/MyDrive/dataset/train"
  val_folder = "/content/drive/MyDrive/dataset/val"
  test_folder = "/content/drive/MyDrive/dataset/test"

  split_dataset(source_folder, train_folder, val_folder, test_folder)
