from face_mask_fusion_api import FaceMaskFusionAPI
import cv2
import base64


def test_face_mask_fusion():
	# 创建 API 实例
	api = FaceMaskFusionAPI()

	# 读取测试图片
	with open('source_pic.jpeg', 'rb') as f:
		source_data = f.read()
	with open('source_pic.jpeg', 'rb') as f:
		mask_data = f.read()

	# 设置混合因子（0-1之间）
	blend_factor = 0.7

	# 执行融合
	result = api.fusion_face_mask(source_data, mask_data, blend_factor)

	if result.success:
		# 保存结果
		img_data = base64.b64decode(result.image_base64)
		with open('result.jpg', 'wb') as f:
			f.write(img_data)
		print(f"成功: {result.message}")
	else:
		print(f"失败: {result.message}")


if __name__ == '__main__':
	test_face_mask_fusion()
