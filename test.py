# # test_api.py
# import requests
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import facefusion
from facefusion import state_manager, processors
from facefusion.processors.modules import face_swapper, face_editor, face_enhancer
import gradio as gr

def setup_facefusion():
	# 加载源数据（人脸图像）
	source_image_paths = ['C:/Users/yangwei/Desktop/source_pic.jpeg']  # 替换为实际的人脸图像路径
	state_manager.set_item('source_paths', source_image_paths)

	# 加载目标数据（脸谱图像）
	target_image_paths = ['C:/Users/yangwei/Desktop/target_pic.jpeg']  # 替换为实际的脸谱图像路径
	state_manager.set_item('target_paths', target_image_paths)

def swap_faces():
    # 执行脸部交换
    face_swapper.pre_check()
    face_swapper.render()
    face_swapper.listen()
    face_swapper.run()

def edit_faces():
    # 编辑脸部图像
    face_editor.pre_check()
    face_editor.render()
    face_editor.listen()
    face_editor.run()

def enhance_faces():
    # 增强脸部图像
    face_enhancer.pre_check()
    face_enhancer.render()
    face_enhancer.listen()
    face_enhancer.run()

def main():
    setup_facefusion()  # 配置FaceFusion
    swap_faces()  # 执行脸部交换
    edit_faces()  # 编辑脸部图像
    enhance_faces()  # 增强脸部图像

    # 输出处理后的图像
    output_paths = state_manager.get_item('output_paths')
    if output_paths:
        for path in output_paths:
            print(f'Processed image saved at: {path}')
    else:
        print('No processed image found.')

if __name__ == '__main__':
    main()


# def test_face_swap():
# 	url = "http://localhost:8000/swap-face/"
#
# 	try:
# 		# 准备文件
# 		files = {
# 			'source_image': ('source.jpg', open('C:/Users/yangwei/Desktop/source_pic.jpeg', 'rb'), 'image/jpeg'),
# 			'target_image': ('target.jpg', open('C:/Users/yangwei/Desktop/target_pic.jpeg', 'rb'), 'image/jpeg')
# 		}
#
# 		# 发送请求
# 		response = requests.post(url, files=files)
#
# 		# 检查响应
# 		if response.status_code == 200:
# 			# 保存结果图片
# 			with open('result.jpg', 'wb') as f:
# 				f.write(response.content)
# 			logger.info("Face swap completed successfully")
# 		else:
# 			logger.error(f"Error: {response.status_code}")
# 			logger.error(response.json())
#
# 	except Exception as e:
# 		logger.error(f"Test failed: {str(e)}")
#
#
# if __name__ == "__main__":
# 	test_face_swap()
