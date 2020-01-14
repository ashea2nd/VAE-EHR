import configparser
import json
import arrow

class Config:
	def __init__(self, config_file_path: str):
		
		config = configparser.ConfigParser()
		config.read(config_file_path)
		
		###DATA Settings
		#Path to Patient_ICD and ICD9Code CSV's
		self.patient_icd_path = config['EXPERIMENT']['patient_icd_path']
		self.icd9codes_path = config['EXPERIMENT']['icd9codes_path']
		
		#Experiment Name
		date = arrow.now().format('YYMMDD')
		experiment = config['EXPERIMENT']['experiment']
		self.experiment_name = "{}_{}".format(date, experiment)

		###MODEL Settings
		#Encoder 
		self.encoder_dim = []
		encoder_dim = json.loads(config['MODEL SETTINGS']['encoder_dim'])
		if len(encoder_dim) > 0:
			self.encoder_dim = [ (encoder_dim[i], encoder_dim[i+1]) for i in range(len(encoder_dim) - 1) ]

		#Latent Dimension
		self.latent_dim = int(config['MODEL SETTINGS']['latent_dim'])

		#Decoder Dimensions
		self.decoder_dim = []
		decoder_dim = json.loads(config['MODEL SETTINGS']['decoder_dim'])
		if len(decoder_dim) > 0:
			self.decoder_dim = [ (decoder_dim[i], decoder_dim[i+1]) for i in range(len(decoder_dim) - 1) ]

		#Use Relu in Encoder
		self.use_relu = bool(config['MODEL SETTINGS']['use_relu'])

		###TRAINER SETTINGS
		self.kld_beta = float(config['TRAINER SETTINGS']['kld_beta'])