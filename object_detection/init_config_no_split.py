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
flags.DEFINE_string('cont_test', '4', 'number of images in test folder.')
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
	new_line = "  num_examples: " + FLAGS.cont_test
	change_line(detector_config, 182, new_line)

if __name__ == '__main__':
  tf.app.run()




