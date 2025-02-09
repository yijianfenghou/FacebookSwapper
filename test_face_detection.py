from face_detection import FaceDetector
import cv2


def test_face_detection(image_path: str):
	"""测试人脸检测"""
	try:
		# 1. 创建检测器实例
		detector = FaceDetector()
		print("人脸检测器创建成功")

		# 2. 读取图像
		image = detector.read_image(image_path)
		if image is None:
			print("图像读取失败")
			return

		# 3. 执行人脸检测
		success, faces, message = detector.detect_faces(image)

		# 4. 处理检测结果
		print(f"检测结果: {message}")

		if success:
			print(f"检测到 {len(faces)} 个人脸:")

			# 5. 在图像上标记人脸（可选）
			result_image = image.copy()
			for i, face in enumerate(faces):
				# 获取人脸框
				x1, y1, x2, y2 = face.bbox.astype(int)

				# 绘制边界框
				cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

				# 添加标签
				cv2.putText(result_image,
							f"Face {i + 1}",
							(x1, y1 - 10),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.9,
							(0, 255, 0),
							2)

				# 如果有关键点，绘制关键点
				if hasattr(face, 'landmark') and face.landmark is not None:
					landmarks = face.landmark.astype(int)
					for point in landmarks:
						cv2.circle(result_image, tuple(point), 2, (0, 0, 255), -1)

			# 6. 保存结果图像
			output_path = 'detected_faces.jpg'
			cv2.imwrite(output_path, result_image)
			print(f"结果已保存至: {output_path}")

		else:
			print("未检测到人脸或检测失败")

	except Exception as e:
		print(f"测试过程出错: {str(e)}")


if __name__ == "__main__":
	# 测试图像路径
	image_path = "source.jpg"
	test_face_detection(image_path)
