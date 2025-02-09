from typing import Optional, List, Tuple
import cv2
import numpy as np
import base64
from dataclasses import dataclass
import torch
from facefusion import state_manager
from facefusion.face_analyser import get_many_faces
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.processors.modules import face_swapper
from facefusion.typing import VisionFrame, Face
from facefusion.content_analyser import clear_inference_pool


@dataclass
class FaceMaskFusionResult:
	success: bool
	message: str
	image_base64: Optional[str] = None
	faces_count: int = 0


class FaceMaskFusionAPI:
	def __init__(self):
		self._init_state()

	def _init_state(self) -> None:
		execution_providers = []
		if torch.cuda.is_available():
			execution_providers.append('CUDAExecutionProvider')
		execution_providers.append('CPUExecutionProvider')

		# 设置执行提供程序
		state_manager.set_item('execution_providers', execution_providers)

		# 设置人脸检测器参数
		state_manager.set_item('face_detector_model', 'yoloface')
		state_manager.set_item('face_detector_size', '640x640')
		state_manager.set_item('face_detector_score', 0.5)
		state_manager.set_item('face_detector_angles', [0])

		print("初始化完成")
		print(f"执行提供程序: {execution_providers}")

	def _read_cv2_image(self, image_data: bytes) -> Optional[np.ndarray]:
		try:
			nparr = np.frombuffer(image_data, np.uint8)
			image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

			if image is None:
				raise ValueError("图像解码失败")

			print(f"读取图像成功，形状: {image.shape}")
			return image

		except Exception as e:
			print(f"读取图片错误: {str(e)}")
			return None

	def _ensure_3_channels(self, image: np.ndarray) -> np.ndarray:
		if image is None:
			raise ValueError("输入图像为空")

		if len(image.shape) == 2:
			return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		elif len(image.shape) == 3:
			if image.shape[2] == 1:
				return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			elif image.shape[2] == 3:
				return image
			elif image.shape[2] == 4:
				return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

		raise ValueError(f"不支持的图像形状: {image.shape}")

	def _detect_faces(self, image: np.ndarray) -> List[Face]:
		"""检测图片中的所有人脸"""
		try:
			if image is None:
				print("输入图像为空")
				return []

			# 确保图像是3通道的BGR格式
			image = self._ensure_3_channels(image)
			print(f"处理图像形状: {image.shape}")

			# 将单个图像包装为列表
			vision_frames = [image]

			# 调用人脸检测
			faces = get_many_faces(vision_frames)

			# 处理返回结果
			if faces is None:
				print("人脸检测返回None")
				return []

			if not faces:
				print("未检测到人脸")
				return []

			print(f"检测到 {len(faces)} 个人脸")
			return faces

		except Exception as e:
			print(f"人脸检测错误: {str(e)}")
			print(f"图像信息: shape={image.shape if image is not None else 'None'}")
			return []

	def _create_face_mask(self, face: Face, image_shape: Tuple[int, int]) -> np.ndarray:
		"""创建人脸区域的蒙版"""
		try:
			# 创建单通道mask
			mask = np.zeros(image_shape[:2], dtype=np.float32)

			if hasattr(face, 'landmark') and face.landmark is not None:
				# 使用人脸关键点创建蒙版
				landmarks = face.landmark.astype(np.int32)
				hull = cv2.convexHull(landmarks)
				cv2.fillConvexPoly(mask, hull, 1.0)

				# 平滑蒙版边缘
				mask = cv2.GaussianBlur(mask, (21, 21), 11)

			return mask

		except Exception as e:
			print(f"创建面部蒙版错误: {str(e)}")
			return np.zeros(image_shape[:2], dtype=np.float32)

	def _blend_images(self,
					  original: np.ndarray,
					  mask_image: np.ndarray,
					  alpha_mask: np.ndarray,
					  blend_factor: float = 0.7) -> np.ndarray:
		"""混合原始图片和脸谱"""
		try:
			# 确保所有输入都是正确的形状
			original = self._ensure_3_channels(original)
			mask_image = self._ensure_3_channels(mask_image)

			# 确保 alpha_mask 是单通道
			if len(alpha_mask.shape) == 3:
				alpha_mask = alpha_mask[:, :, 0]

			# 数据类型转换和归一化
			original = original.astype(np.float32) / 255.0
			mask_image = mask_image.astype(np.float32) / 255.0
			alpha_mask = alpha_mask.astype(np.float32)

			# 扩展 alpha_mask 到3通道
			alpha = np.dstack([alpha_mask] * 3) * blend_factor

			# 执行混合
			result = (1.0 - alpha) * original + alpha * mask_image

			# 转换回uint8
			result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
			return result

		except Exception as e:
			print(f"混合图像时出错: {str(e)}")
			print(f"形状信息:")
			print(f"Original: {original.shape if original is not None else 'None'}")
			print(f"Mask: {mask_image.shape if mask_image is not None else 'None'}")
			print(f"Alpha: {alpha_mask.shape if alpha_mask is not None else 'None'}")
			return original

	def fusion_face_mask(self,
						 source_image_data: bytes,
						 mask_image_data: bytes,
						 blend_factor: float = 0.7) -> FaceMaskFusionResult:
		"""融合脸谱到人脸图片中"""
		try:
			# 读取图片
			source_image = self._read_cv2_image(source_image_data)
			mask_image = self._read_cv2_image(mask_image_data)

			if source_image is None or mask_image is None:
				return FaceMaskFusionResult(success=False, message="无法读取输入图片")

			print("开始人脸检测...")
			# 检测源图片中的人脸
			faces = self._detect_faces(source_image)

			if not faces:
				return FaceMaskFusionResult(success=False, message="未检测到人脸")

			print(f"检测到 {len(faces)} 个人脸，开始处理...")

			# 处理每个人脸
			result_image = source_image.copy()
			faces_processed = 0

			for face in faces:
				try:
					# 调整脸谱大小并对齐到人脸位置
					aligned_mask = self._resize_and_align_mask(
						mask_image,
						face,
						source_image.shape
					)

					# 创建人脸区域的蒙版
					face_mask = self._create_face_mask(face, source_image.shape[:2])

					# 混合图像
					result_image = self._blend_images(
						result_image,
						aligned_mask,
						face_mask,
						blend_factor
					)

					faces_processed += 1
					print(f"处理完成第 {faces_processed} 个人脸")

				except Exception as e:
					print(f"处理第 {faces_processed + 1} 个人脸时出错: {str(e)}")
					continue

			if faces_processed == 0:
				return FaceMaskFusionResult(
					success=False,
					message="所有人脸处理均失败",
					faces_count=len(faces)
				)

			# 编码结果图片
			print("编码结果图片...")
			success, buffer = cv2.imencode('.jpg', result_image)
			if not success:
				return FaceMaskFusionResult(
					success=False,
					message="结果图片编码失败",
					faces_count=len(faces)
				)

			result_base64 = base64.b64encode(buffer).decode('utf-8')

			return FaceMaskFusionResult(
				success=True,
				message=f"成功处理 {faces_processed}/{len(faces)} 个人脸",
				image_base64=result_base64,
				faces_count=len(faces)
			)

		except Exception as e:
			print(f"处理过程出错: {str(e)}")
			return FaceMaskFusionResult(
				success=False,
				message=f"处理过程出错: {str(e)}",
				faces_count=0
			)
		finally:
			clear_inference_pool()

	def _resize_and_align_mask(self,
							   mask_image: np.ndarray,
							   face: Face,
							   target_shape: Tuple[int, int, int]) -> np.ndarray:
		"""调整脸谱大小并对齐到人脸位置"""
		try:
			mask_image = self._ensure_3_channels(mask_image)

			x1, y1, x2, y2 = face.bbox.astype(int)
			face_width = x2 - x1
			face_height = y2 - y1

			# 计算缩放比例
			scale = min(face_width / mask_image.shape[1],
						face_height / mask_image.shape[0])

			# 确保新尺寸至少为1
			new_width = max(1, int(mask_image.shape[1] * scale))
			new_height = max(1, int(mask_image.shape[0] * scale))

			# 缩放图像
			resized_mask = cv2.resize(mask_image, (new_width, new_height))

			# 创建目标大小的空白图像
			aligned_mask = np.zeros(target_shape, dtype=np.uint8)

			# 计算居中位置
			x_offset = x1 + (face_width - new_width) // 2
			y_offset = y1 + (face_height - new_height) // 2

			# 确保偏移量不会超出图像边界
			x_offset = max(0, min(x_offset, target_shape[1] - new_width))
			y_offset = max(0, min(y_offset, target_shape[0] - new_height))

			# 将缩放后的脸谱放置到对应位置
			aligned_mask[y_offset:y_offset + new_height,
			x_offset:x_offset + new_width] = resized_mask

			return aligned_mask

		except Exception as e:
			print(f"调整脸谱大小和位置时出错: {str(e)}")
			return np.zeros(target_shape, dtype=np.uint8)
