# Object_Detection
Código para treino e inferência de Detecção e Classifcação de objetos. Vídeos de demonstração dos resultados baseados em um experimento de detecção de disco e pinça de freio:
- Vídeo da base de dados utilizada: 
- Vídeo de teste na esteira:
- Vídeo de teste em ambiente não controlado: https://youtu.be/JFh3peJrghI

# Organizar base de dados:
## Instalação e preparação
	- Instalar dependências (PyQt, tensorflow, e outras)
	- Instalar labelImg
	- Criar base de dados com xml e jpg com mesmo nome, usando labelImg
	- Instalar API do object_detection utilizando o setup.py
	- Separar, no diretório 'imagens', em 'test' e 'train'

## Configuração inicial
	- Alterar o arquivo 'data/classes.txt' com os mesmos nomes das classes definidas no xml. Cada nome de classe deve estar em uma linha do txt
	- Rodar o código 'python init_config.py' da pasta 'object_detection'
	- Baixar arquivos de rede pré treinada ssd_mobilenet_v1_coco_11_06_2017, colocar arquivos na pasta  'Detector'. Para usar sua própria rede para continuar, substituia os arquivos pelos gerados ao exportar o grafo de inferencia.
	- Editar 'num_steps' na linha 164 do arquivo 'training/detector.config'


# Rodar treino (NO DIRETÓRIO object_detection):
## Rodar código 'xml_to_csv.py' no diretório object_detection
	- python xml_to_csv.py
	-- criará os arquivos .csv relacionando a imagem .jpg às informações do .xml	

## Rodar código 'generate_tfrecord.py'
	- cd ..
	- export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	- cd object_detection	
	- python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
	- python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
	-- criará arquivos .record no diretório 'data'

## Rodar código 'train.py'
	- cd ..
	- export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	- cd object_detection	
	- python3 train.py --logtostderr --train_dir=train_log/ --pipeline_config_path=training/detector.config


# Exportar grafo de inferencia
	- cd ..
	- export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	- cd object_detection	
	- python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/detector.config --trained_checkpoint_prefix train_log/model.ckpt-XXXX --output_directory inference_graph