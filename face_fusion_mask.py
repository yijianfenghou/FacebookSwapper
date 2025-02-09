from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from rembg import remove
from ultralytics import YOLO
import dlib
import torch

from facefusion.processors.modules import face_swapper
from facefusion.processors.modules import face_editor
from facefusion.processors.modules import face_enhancer
from facefusion.face_store import Face
from facefusion import state_manager

# 定义类型
VisionFrame = np.ndarray


class FaceMaskProcessor:
	def __init__(self):
		self.initialization_time = '2025-02-08 14:33:57'
		self.initialized_by = 'yijianfenghou'

		# 初始化模型
		self.face_detector = YOLO('models/yolov11n-face.pt')
		self.landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

		self._initialize_state()

	def _initialize_state(self):
		try:
			state_manager.set_item('face_swapper_model', 'blendswap_256')
			state_manager.set_item('face_swapper_threshold', 0.7)
			state_manager.set_item('face_editor_model', 'gfpgan_1.4')
			state_manager.set_item('face_editor_blend', 80)
			state_manager.set_item('face_enhancer_model', 'real_esrgan_x4')
			state_manager.set_item('face_enhancer_blend', 100)
		except Exception as e:
			print(f"Error in state initialization: {str(e)}")
			raise

	def create_face(self, bbox: np.ndarray, landmarks: np.ndarray = None, confidence: float = 1.0) -> Face:
		"""创建Face对象的辅助方法"""
		return Face(
			bounding_box=bbox,
			score_set=[confidence],
			landmark_set=[landmarks] if landmarks is not None else [None],
			angle=0.0,
			embedding=np.zeros(512),  # 默认embedding大小
			normed_embedding=np.zeros(512),  # 默认normalized embedding大小
			gender=0,  # 0: unknown, 1: male, 2: female
			age=0,  # 默认年龄
			race=0  # 默认种族
		)

	def detect_faces(self, image: VisionFrame) -> List[Face]:
		"""使用YOLOv8检测人脸并使用dlib获取关键点"""
		try:
			yolo_results = self.face_detector(image, conf=0.5, iou=0.7)

			if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
				return []

			faces = []
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			for box in yolo_results[0].boxes:
				try:
					bbox = box.xyxy[0].cpu().numpy()
					conf = float(box.conf[0])

					dlib_rect = dlib.rectangle(
						left=int(bbox[0]),
						top=int(bbox[1]),
						right=int(bbox[2]),
						bottom=int(bbox[3])
					)
					landmarks = self.get_landmarks_from_dlib(gray, dlib_rect)

					if landmarks is not None:
						face = self.create_face(bbox, landmarks, conf)
						faces.append(face)

				except Exception as e:
					print(f"Error processing individual face: {str(e)}")
					continue

			return faces

		except Exception as e:
			print(f"Error in face detection: {str(e)}")
			return []

	def get_landmarks_from_dlib(self, image: np.ndarray, rect) -> np.ndarray:
		"""使用dlib获取人脸关键点"""
		try:
			shape = self.landmark_predictor(image, rect)
			return np.array([[p.x, p.y] for p in shape.parts()])
		except Exception as e:
			print(f"Error getting landmarks: {str(e)}")
			return None

	def process_face(self, source_frame: VisionFrame, target_frame: VisionFrame, face_num: int = 1) -> VisionFrame:
		"""处理人脸替换为脸谱的主要函数"""
		try:
			target_faces = self.detect_faces(target_frame)
			if not target_faces:
				raise ValueError("No face detected in target image")

			target_faces = target_faces[:face_num]

			# 创建source face
			source_face = self.create_face(
				bbox=np.array([0, 0, source_frame.shape[1], source_frame.shape[0]])
			)

			# 处理步骤
			swapped = face_swapper.process_frame({
				'source_face': source_face,
				'target_faces': target_faces,
				'source_frame': source_frame,
				'target_frame': target_frame
			})

			edited = face_editor.process_frame({
				'faces': target_faces,
				'frame': swapped
			})

			enhanced = face_enhancer.process_frame({
				'faces': target_faces,
				'frame': edited
			})

			return enhanced

		except Exception as e:
			print(f"Error in face processing: {str(e)}")
			return target_frame

	def process_mask_image(self, mask_image: VisionFrame) -> VisionFrame:
		"""处理脸谱图片"""
		try:
			mask_pil = Image.fromarray(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
			mask_no_bg = remove(mask_pil)
			mask_cv = cv2.cvtColor(np.array(mask_no_bg), cv2.COLOR_RGBA2BGR)
			return cv2.resize(mask_cv, (512, 512))
		except Exception as e:
			print(f"Error processing mask: {str(e)}")
			return mask_image

	def run(self, mask_path: str, face_path: str, output_path: str, face_num: int = 1) -> None:
		"""运行完整的处理流程"""
		try:
			print(f"Starting processing at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

			mask_image = cv2.imread(mask_path)
			face_image = cv2.imread(face_path)

			if mask_image is None or face_image is None:
				raise ValueError("Cannot read input images")

			processed_mask = self.process_mask_image(mask_image)
			result = self.process_face(processed_mask, face_image, face_num)

			cv2.imwrite(output_path, result)
			print(f"Successfully saved result to {output_path}")

		except Exception as e:
			print(f"Error in processing: {str(e)}")
			raise


def create_processor_with_config():
	"""创建处理器实例并返回配置信息"""
	processor = FaceMaskProcessor()

	# 返回当前配置信息
	config = {
		'initialization_time': processor.initialization_time,
		'initialized_by': processor.initialized_by,
		'face_detector': {
			'model': 'yolov8',
			'confidence': 0.5,
			'iou': 0.7
		},
		'face_swapper': {
			'model': state_manager.get_item('face_swapper_model'),
			'threshold': state_manager.get_item('face_swapper_threshold')
		},
		'face_editor': {
			'model': state_manager.get_item('face_editor_model'),
			'blend': state_manager.get_item('face_editor_blend')
		},
		'face_enhancer': {
			'model': state_manager.get_item('face_enhancer_model'),
			'blend': state_manager.get_item('face_enhancer_blend')
		}
	}

	return processor, config


def create_processor():
	return FaceMaskProcessor()
