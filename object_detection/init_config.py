# -*- coding: utf-8 -*-
"""
Spyder Editor

This code edit the files:
	- object-detection.pbtxt (create item{ id; name; })
	- generate_tfrecord.py (if conditions)
	- detector.config (num_classes)

Input:
	- data/classes.txt: file with name of classes separeted by new line (each class name per line).

"""

import os, glob, shutil
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('batch_size', '4', 'batch size for training.')
FLAGS = flags.FLAGS

def change_line(path,index_linha,nova_linha):
    with open(path,'r') as f:
        texto=f.readlines()
    with open(path,'w') as f:
        for i in texto:
            if texto.index(i)==index_linha:
                f.write(nova_linha+'\n')
            else:
                f.write(i)

def main(_):
	# SPLIT TRAIN/TEST IMAGE FOLDERS
	local_files = os.getcwd()
	source = local_files + "/images/resized"
	test_path = local_files + "/images/test"
	train_path = local_files + "/images/train"
	test_img_path = local_files + "/test_images"

	dire = os.listdir(test_path)
	for arq in dire:
		arq = test_path + "/" +  arq
		os.remove(arq)

	dire = os.listdir(train_path)
	for arq in dire:
		arq = train_path + "/" +  arq
		os.remove(arq)

	dire = os.listdir(test_img_path)
	for arq in dire:
		arq = test_img_path + "/" +  arq
		os.remove(arq)

	cont = 0
	cont_t=0
	cont_test=0
	os.chdir(local_files)
	for f in glob.glob(source + "/*.xml"):
		cont = cont + 1
		if not cont%10: #10% used for tests
			new_path = test_path + "/test_" + str(int(cont/10)) + ".jpg"
			shutil.copyfile(f[:-4] + ".jpg", new_path)
			new_path = test_path + "/test_" + str(int(cont/10)) + ".xml"
			shutil.copyfile(f,new_path)
			new_path = test_img_path + "/test_" + str(int(cont/10)) + ".jpg"
			shutil.copyfile(f[:-4] + ".jpg",new_path)
			cont_test = cont_test + 1
		else:
			cont_t = cont_t + 1
			new_path = train_path + "/train_" + str(cont_t) + ".jpg"
			shutil.copyfile(f[:-4] + ".jpg",str(new_path))
			new_path = train_path + "/train_" + str(cont_t) + ".xml"
			shutil.copyfile(f,new_path)
			#if not cont_t%20: #20% used for test
			new_path = test_img_path + "/train_" + str(cont_t) + ".jpg"
			shutil.copyfile(f[:-4] + ".jpg",str(new_path))
			#cont_test = cont_test + 1
		

	# TXT WITH CLASSES
	local_files = os.getcwd()
	source = local_files + "/data"
	os.chdir(source)
	f_classes = open("classes.txt", 'r')
	#classes = f_classes.readlines()
	classes = f_classes.read().splitlines()
	f_classes.close()

	# OPEN FILES
	obj_det_path = local_files + "/training" # erase and re-write
	gen_tfrec_path = local_files # re-write set of lines
	detec_config_path = local_files + "/training" # re-write single line

	os.chdir(gen_tfrec_path)
	f_generate_tfrecord = open("generate_tfrecord_test.py", "r")
	generate_tfrecord_contents = f_generate_tfrecord.readlines()
	f_generate_tfrecord.close()

	os.chdir(obj_det_path)
	open("object-detection.pbtxt", "w").close
	f_object_detection = open("object-detection.pbtxt", 'a')

	cont = 0
	index = 30
	value = ""
	for label in classes:
		value = "	if row_label == '" + label + "':\n" + "		return " + str(cont+1) + "\n"
		generate_tfrecord_contents.insert(index + cont, value)

		cont = cont + 1
		value = "item {\n  id: " + str(cont) + "\n  name: '" + label + "'\n}\n"
		f_object_detection.write(value)

	f_object_detection.close()

	os.chdir(local_files)
	f_generate_tfrecord = open("generate_tfrecord.py", "w")
	generate_tfrecord_contents = "".join(generate_tfrecord_contents)
	f_generate_tfrecord.write(generate_tfrecord_contents)
	f_generate_tfrecord.close()

	# REWRITE NUM_CLASSES	
	os.chdir(detec_config_path)
	detector_config = detec_config_path+"/detector.config"
	new_line = "    num_classes: " + str(cont)
	change_line(detector_config, 8, new_line)
	new_line = "  batch_size: " + FLAGS.batch_size
	change_line(detector_config, 142, new_line)
	new_line = "  num_examples: " + str(cont_test)
	change_line(detector_config, 182, new_line)

if __name__ == '__main__':
  tf.app.run()




