import tensorflow as tf

# definisci il percorso del file TFRecord
file_path = "data/dataset/train.record"

# crea un dataset dal file TFRecord
dataset = tf.data.TFRecordDataset(file_path)

# conta il numero di record presenti nel dataset
num_images = sum(1 for _ in dataset)

print("Numero di immagini per il set di train : {}".format( num_images))
file_path = "data/dataset/val.record"

# crea un dataset dal file TFRecord
dataset = tf.data.TFRecordDataset(file_path)

# conta il numero di record presenti nel dataset
num_images = sum(1 for _ in dataset)

print("Numero di immagini per il set di validation : {}".format( num_images))
