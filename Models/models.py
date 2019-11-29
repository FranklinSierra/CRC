#librerias

from Models import LeNet, MobilNet, VGG16, VGG19

#creacion de las banderas
parser = OptionParser()
parser.add_option("--dataset", dest="dataset", type=str, 
help="Wich dataset to load", default='WL')
(options, args) = parser.parse_args()
#despues del . va lo de dest en el add_option
dataset_name = options.dataset