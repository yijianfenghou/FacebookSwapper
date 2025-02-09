from datetime import datetime

# 系统配置
SYSTEM_CONFIG = {
	'current_time': '2025-02-08 12:21:13',
	'current_user': 'yijianfenghou',
	'debug_mode': True
}

# 路径配置
PATH_CONFIG = {
	'upload_folder': 'uploads',
	'output_folder': 'output',
	'models_folder': 'models',
	'logs_folder': 'logs',
	'temp_folder': 'temp',
	# 将 set 改为 list
	'allowed_extensions': ['png', 'jpg', 'jpeg']  # 修改这里
}

# 人脸检测配置
FACE_DETECTOR_CONFIG = {
	'angles': [0, -20, 20],
	'model': 'retinaface',
	'size': '640x640',
	'selector_mode': 'many',
	'min_face_size': 30,
	'detection_threshold': 0.3
}

# 人脸替换配置
FACE_SWAPPER_CONFIG = {
	'model': 'blendswap_256',
	'threshold': 0.7
}

# 人脸编辑配置
FACE_EDITOR_CONFIG = {
	'model': 'gfpgan_1.4',
	'blend': 80,
	'regions': ['eyes', 'mouth', 'nose']
}

# 人脸增强配置
FACE_ENHANCER_CONFIG = {
	'model': 'real_esrgan_x4',
	'blend': 100,
	'tile_size': 512,
	'tile_padding': 10
}

# 图像预处理配置
IMAGE_PREPROCESSING_CONFIG = {
	'target_size': [512, 512],  # 将元组改为列表
	'clahe_clip_limit': 3.0,
	'clahe_grid_size': [8, 8],  # 将元组改为列表
	'contrast_alpha': 1.2,
	'brightness_beta': 10,
	'denoise_h': 7,
	'sharpen_kernel': [[-1, -1, -1],
					   [-1, 9, -1],
					   [-1, -1, -1]]
}


def get_all_config():
	"""获取所有配置"""
	return {
		'system': SYSTEM_CONFIG,
		'paths': PATH_CONFIG,
		'face_detector': FACE_DETECTOR_CONFIG,
		'face_swapper': FACE_SWAPPER_CONFIG,
		'face_editor': FACE_EDITOR_CONFIG,
		'face_enhancer': FACE_ENHANCER_CONFIG,
		'image_preprocessing': IMAGE_PREPROCESSING_CONFIG
	}


def update_config(section: str, key: str, value: any) -> bool:
	"""
	更新配置值

	Args:
		section: 配置部分名称
		key: 配置键
		value: 新值

	Returns:
		是否更新成功
	"""
	try:
		config_map = {
			'system': SYSTEM_CONFIG,
			'paths': PATH_CONFIG,
			'face_detector': FACE_DETECTOR_CONFIG,
			'face_swapper': FACE_SWAPPER_CONFIG,
			'face_editor': FACE_EDITOR_CONFIG,
			'face_enhancer': FACE_ENHANCER_CONFIG,
			'image_preprocessing': IMAGE_PREPROCESSING_CONFIG
		}

		if section in config_map and key in config_map[section]:
			config_map[section][key] = value
			return True
		return False

	except Exception as e:
		print(f"Error updating config: {str(e)}")
		return False
