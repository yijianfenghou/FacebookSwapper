from typing import List, Optional, Tuple
import cv2
import numpy as np
from facefusion import state_manager
from facefusion.face_analyser import get_many_faces
from facefusion.typing import VisionFrame, Face


class FaceDetector:
	def __init__(self):
		"""初始化人脸检测器"""
		self._init_state()

	def _init_state(self) -> None:
		"""初始化状态配置"""
		execution_providers = ['CPUExecutionProvider']
		state_manager.set_item('execution_providers', execution_providers)

		# 配置人脸检测参数
		state_manager.set_item('face_detector_model', 'yoloface')
		state_manager.set_item('face_detector_size', '640x640')
		state_manager.set_item('face_detector_score', 0.5)
		state_manager.set_item('face_detector_angles', [0])

		print("人脸检测器初始化完成")

	def _ensure_3_channels(self, image: np.ndarray) -> np.ndarray:
		"""确保图像是3通道的"""
		if image is None:
			raise ValueError("输入图像为空")

		if len(image.shape) == 2:  # 单通道图像
			return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		elif len(image.shape) == 3:
			if image.shape[2] == 1:  # 单通道图像带维度
				return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			elif image.shape[2] == 3:  # 已经是3通道
				return image
			elif image.shape[2] == 4:  # 4通道图像（带alpha通道）
				return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

		raise ValueError(f"不支持的图像形状: {image.shape}")

	def detect_faces(self, image: np.ndarray) -> Tuple[bool, List[Face], str]:
		"""
		检测图像中的人脸

		Args:
			image: 输入图像 (numpy.ndarray)

		Returns:
			Tuple[bool, List[Face], str]:
			- 是否成功检测到人脸
			- 检测到的人脸列表
			- 状态消息
		"""
		try:
			# 1. 验证输入
			if image is None:
				return False, [], "输入图像为空"

			# 2. 确保图像是3通道的
			try:
				image = self._ensure_3_channels(image)
			except ValueError as e:
				return False, [], f"图像格式错误: {str(e)}"

			# 3. 打印图像信息
			print(f"处理图像形状: {image.shape}")

			# 4. 将图像包装为列表（因为get_many_faces期望接收图像列表）
			vision_frames = [image]

			# 5. 执行人脸检测
			faces = get_many_faces(vision_frames)

			# 6. 处理检测结果
			if faces is None:
				return False, [], "人脸检测失败（返回None）"

			if not faces:
				return False, [], "未检测到人脸"

			# 7. 返回检测结果
			return True, faces, f"成功检测到 {len(faces)} 个人脸"

		except Exception as e:
			error_msg = f"人脸检测出错: {str(e)}"
			print(error_msg)
			if image is not None:
				print(f"图像信息: shape={image.shape}")
			return False, [], error_msg

	def read_image(self, image_path: str) -> Optional[np.ndarray]:
		"""
		读取图像文件

		Args:
			image_path: 图像文件路径

		Returns:
			Optional[np.ndarray]: 读取的图像，失败返回None
		"""
		try:
			image = cv2.imread(image_path)
			if image is None:
				print(f"无法读取图像: {image_path}")
				return None

			print(f"成功读取图像: {image_path}, 形状: {image.shape}")
			return image
		except Exception as e:
			print(f"读取图像出错: {str(e)}")
			return None
