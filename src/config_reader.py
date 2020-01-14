import configparser
import json

class Config:
	def __init__(self, config_file_path: str):
		
		config = configparser.ConfigParser()
		config.read(config_file_path)
		
		self.patient_icd_path = config['EXPERIMENT']['patient_icd_path']
		self.icd9codes_path = config['EXPERIMENT']['icd9codes_path']
		self.experiment = config['EXPERIMENT']['experiment']

		self.encoder_dim = []
		encoder_dim = json.loads(config['MODEL SETTINGS']['encoder_dim'])
		if len(encoder_dim) > 0:
			self.encoder_dim = [ (encoder_dim[i], encoder_dim[i+1]) for i in range(len(encoder_dim) - 1) ]

		self.latent_dim = int(config['MODEL SETTINGS']['latent_dim'])

		self.decoder_dim = []
		decoder_dim = json.loads(config['MODEL SETTINGS']['decoder_dim'])
		if len(decoder_dim) > 0:
			self.decoder_dim = [ (decoder_dim[i], decoder_dim[i+1]) for i in range(len(decoder_dim) - 1) ]

		self.use_relu = bool(config['MODEL SETTINGS']['use_relu'])