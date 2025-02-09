import os
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
import cv2
from face_fusion_mask import create_processor

app = Flask(__name__, static_url_path='/static')

# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# 配置上传文件存储路径 - 放在static目录下
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(STATIC_DIR, 'results')  # 修改为 results
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 创建必要的文件夹
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # 修改为 results

# 输出初始化信息
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: yijianfenghou")
print(f"Static directory: {STATIC_DIR}")
print(f"Upload directory: {UPLOAD_FOLDER}")
print(f"Results directory: {RESULTS_FOLDER}")  # 修改为 results

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    try:
        # 检查是否有文件上传
        if 'mask' not in request.files or 'face' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        mask_file = request.files['mask']
        face_file = request.files['face']

        # 检查文件名是否合法
        if not (mask_file.filename and face_file.filename):
            return jsonify({'error': 'No selected files'}), 400

        if not (allowed_file(mask_file.filename) and allowed_file(face_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400

        # 获取处理的人脸数量
        face_num = int(request.form.get('face_num', 1))

        # 生成唯一的文件名
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        mask_path = os.path.join(UPLOAD_FOLDER, f'mask_{timestamp}.jpg')
        face_path = os.path.join(UPLOAD_FOLDER, f'face_{timestamp}.jpg')
        result_path = os.path.join(RESULTS_FOLDER, f'result_{timestamp}.jpg')  # 修改为 results

        print(f"Processing files at {timestamp}:")
        print(f"Mask path: {mask_path}")
        print(f"Face path: {face_path}")
        print(f"Result path: {result_path}")  # 修改为 result

        # 保存上传的文件
        mask_file.save(mask_path)
        face_file.save(face_path)

        # 创建处理器并处理图像
        processor = create_processor()
        processor.run(mask_path, face_path, result_path, face_num)  # 修改输出路径

        # 返回结果图片的URL
        result_url = f'/static/results/result_{timestamp}.jpg'  # 修改为 results


        print(f"image save successs!!!!")
        return jsonify({
            'success': True,
            'result_url': result_url
        })

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # 清理临时文件
        try:
            if 'mask_path' in locals():
                os.remove(mask_path)
            if 'face_path' in locals():
                os.remove(face_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0',
        'user': 'yijianfenghou'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
