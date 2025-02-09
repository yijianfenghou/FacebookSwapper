from face_fusion_mask import create_processor_with_config
import json


def main():
	# 创建处理器并获取配置
	processor, config = create_processor_with_config()

	# 打印当前配置
	# print("Current configuration:")
	# print(json.dumps(config, indent=2))

	# 设置输入输出路径
	mask_path = "./target_pic.jpeg"  # 脸谱图像
	face_path = "./source.jpg"  # 人脸图像
	output_path = "./output/output.jpg"  # 输出路径

	try:
		# 执行处理
		processor.run(
			mask_path=mask_path,
			face_path=face_path,
			output_path=output_path,
			face_num=1  # 处理单张人脸
		)
		print("Processing completed successfully!")

	except Exception as e:
		print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
	main()
