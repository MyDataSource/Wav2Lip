from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
from scipy.signal import butter, filtfilt

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--mouth_enhance', default=1.0, type=float,
					help='嘴部区域锐度增强因子(1.0=正常, 1.5=更锐利, 0.5=更平滑)')
parser.add_argument('--mouth_sharpen', type=float, default=1.0, 
                    help='口型锐化因子。1.0表示不锐化，大于1的值会增加锐化程度')
parser.add_argument('--lip_saturation', type=float, default=1.0, 
                    help='唇部饱和度提升。1.0表示不变，大于1的值会增加唇部颜色鲜艳度')
parser.add_argument('--image_mode', action='store_true', default=False,
                    help='启用简化的图像处理模式。处理静态图像时使用直接替换方法，没有复杂的混合处理')
parser.add_argument('--color_match', action='store_true', default=False,
                    help='启用严格的颜色匹配，确保生成的视频与原始图片颜色一致')
parser.add_argument('--color_match_mode', type=str, default='histogram', choices=['simple', 'histogram', 'reinhard'],
                    help='颜色匹配算法: simple=简单统计匹配, histogram=直方图匹配, reinhard=Reinhard色调映射')
parser.add_argument('--video_color_preserve', action='store_true', default=False,
                    help='视频模式下也启用颜色保持，确保视频输出颜色与原始视频一致')
parser.add_argument('--color_strength', type=float, default=1.0,
                    help='颜色匹配强度，0.0表示不匹配，1.0表示完全匹配')
parser.add_argument('--advanced_color', action='store_true', default=False,
                    help='启用高级颜色处理，包括多阶段色彩校正和局部颜色保持，对静态图像特别有效')
parser.add_argument('--global_color', action='store_true', default=False,
                    help='启用全局颜色校正，将处理整个图像而不仅仅是面部区域，彻底消除色差')
parser.add_argument('--color_preserve', type=float, default=0.9,
                    help='色彩保留强度，值越高越保留原图颜色 (0.0-1.0)，默认0.9表示保留90%原图色调')
parser.add_argument('--mouth_only', action='store_true', default=False,
                    help='仅改变嘴部区域，完全保持其他区域不变，彻底解决色差问题')

args = parser.parse_args()
args.img_size = 96  # 固定为96，原始模型仅支持这个尺寸
print(f"Using fixed image size of {args.img_size}x{args.img_size} for face crops (model limitation)")

# 检测是否是图片输入
if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
	args.static = True
	print(f"静态图片输入检测到: {args.face}，设置static=True")
	if args.image_mode:
		print(f"启用简化图像处理模式，将使用直接替换方法")

# 选择设备
if torch.cuda.is_available():
    device = 'cuda'
    print('使用CUDA进行推理')
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print('使用MPS (Metal Performance Shaders) 进行推理')
else:
    device = 'cpu'
    print('使用CPU进行推理')

# 初始化临时目录
os.makedirs('temp', exist_ok=True)

# 设置mel步长全局变量
mel_step_size = 16

def get_smoothened_boxes(boxes, T):
    """
    使用更高级的平滑算法来防止面部检测的抖动
    """
    # 原始的简单均值平滑
    smoothed_boxes = boxes.copy()
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
            
        # 使用加权平均，中心帧权重最高
        weights = np.array([max(1.0, T - abs(T//2 - j)) for j in range(len(window))])
        weights = weights / weights.sum()
        
        # 计算加权平均
        weighted_avg = np.zeros(4)
        for j, box in enumerate(window):
            weighted_avg += box * weights[j]
            
        smoothed_boxes[i] = weighted_avg
    
    # 额外的时间一致性平滑（低通滤波）
    if len(boxes) > 2:
        filtered_boxes = smoothed_boxes.copy()
        alpha = 0.25  # 平滑因子
        for i in range(1, len(smoothed_boxes)):
            filtered_boxes[i] = alpha * smoothed_boxes[i] + (1 - alpha) * filtered_boxes[i-1]
        return filtered_boxes
    else:
        return smoothed_boxes

def face_detect(images):
    print("Starting face detection on {} images using device: {}".format(len(images), device))
    
    # 增强图像预处理
    enhanced_images = []
    for img in images:
        # 转换为灰度进行处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 应用CLAHE（对比度受限自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # 锐化图像以突出面部特征
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(img, -1, kernel)
        enhanced_images.append(sharp)
    
    # 使用增强的图像进行人脸检测
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    print("Face detector initialized")

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(enhanced_images), batch_size)):
                # 如果图像太大，先缩小进行检测，然后重新映射坐标
                current_batch_orig = enhanced_images[i:i + batch_size]
                current_batch = []
                resize_factors = []
                
                for img in current_batch_orig:
                    h, w = img.shape[:2]
                    # 如果图像太大，临时缩小以便检测
                    if max(h, w) > 1280:
                        resize_factor = 1280 / max(h, w)
                        resized_img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))
                        current_batch.append(resized_img)
                        resize_factors.append(resize_factor)
                    else:
                        current_batch.append(img)
                        resize_factors.append(1.0)
                
                current_batch = np.array(current_batch)
                print("Processing batch {}/{} with shape: {}".format(
                    i//batch_size + 1, 
                    (len(enhanced_images) + batch_size - 1)//batch_size,
                    current_batch.shape))
                batch_predictions = detector.get_detections_for_batch(current_batch)
                
                # 调整回原始坐标
                for j, (pred, factor) in enumerate(zip(batch_predictions, resize_factors)):
                    if pred is not None and factor != 1.0:
                        # 反向映射回原始分辨率
                        pred[0] = int(pred[0] / factor)
                        pred[1] = int(pred[1] / factor)
                        pred[2] = int(pred[2] / factor)
                        pred[3] = int(pred[3] / factor)
                        batch_predictions[j] = pred
                        
                predictions.extend(batch_predictions)
                print("Batch {}/{} processed, found {} faces".format(
                    i//batch_size + 1,
                    (len(enhanced_images) + batch_size - 1)//batch_size,
                    sum(1 for p in batch_predictions if p is not None)))
        except RuntimeError as e:
            if batch_size == 1: 
                print("Fatal error during face detection: {}".format(e))
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    
    print("Face detection completed, processing results...")

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            print("Running face detection on video frames...")
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            print("Running face detection on static image...")
            face_det_results = face_detect([frames[0]])
            # 对于静态图像，需要复制结果以匹配音频长度
            face_det_results = [face_det_results[0]] * len(mels)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        if args.static:
            face = frames[0][y1: y2, x1:x2]
            face_det_results = [[face, (y1, y2, x1, x2)]] * len(mels)
        else:
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[i % len(face_det_results)].copy()

        # 使用高质量的插值算法调整大小，保留更多细节
        orig_size = face.shape[:2]
        face = cv2.resize(face, (args.img_size, args.img_size), interpolation=cv2.INTER_LANCZOS4)
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

def _load(checkpoint_path):
    print('Loading checkpoint from: {} to device: {}'.format(checkpoint_path, device))
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
        print('Checkpoint loaded with map_location to CPU first')
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Loading checkpoint from: {}".format(path))
    checkpoint = _load(path)
    print("Checkpoint keys: {}".format(list(checkpoint.keys())))
    s = checkpoint["state_dict"]
    print("State dict contains {} keys".format(len(s.keys())))
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    print("Moving model to device: {}".format(device))
    model.load_state_dict(new_s)
    
    model = model.to(device)
    print("Model successfully loaded and moved to {}".format(device))
    return model.eval()

def color_match(source, target, mode='simple', strength=1.0):
    """高级颜色匹配函数，支持多种匹配算法
    
    Args:
        source: 源图像（要调整颜色的图像）
        target: 目标图像（提供颜色分布的参考图像）
        mode: 颜色匹配模式，支持'simple'、'histogram'和'reinhard'
        strength: 颜色匹配强度，0.0表示不匹配，1.0表示完全匹配
        
    Returns:
        颜色匹配后的源图像
    """
    if strength <= 0.0:
        return source.copy().astype(np.uint8)
        
    source = source.copy().astype(np.float32)
    target = target.copy().astype(np.float32)
    
    # 确保图像大小一致
    if source.shape != target.shape:
        target = cv2.resize(target, (source.shape[1], source.shape[0]))
    
    # 如果不需要完全匹配，则准备原始源图像用于混合
    orig_source = source.copy() if strength < 1.0 else None
    
    if mode == 'simple':
        # 简单的颜色统计匹配
        matched = np.zeros_like(source)
        for c in range(3):
            src_mean, src_std = np.mean(source[:,:,c]), np.std(source[:,:,c])
            tgt_mean, tgt_std = np.mean(target[:,:,c]), np.std(target[:,:,c])
            
            if src_std == 0:
                src_std = 1.0
                
            # 标准化并重新缩放
            matched[:,:,c] = ((source[:,:,c] - src_mean) / src_std * tgt_std + tgt_mean)
            
        # 裁剪到有效范围
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        
    elif mode == 'histogram':
        # 直方图匹配 - 分别在各通道上进行
        matched = np.zeros_like(source)
        
        for c in range(3):
            # 计算累积直方图
            src_hist, _ = np.histogram(source[:,:,c].flatten(), 256, [0, 256], density=True)
            tgt_hist, _ = np.histogram(target[:,:,c].flatten(), 256, [0, 256], density=True)
            
            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()
            
            # 归一化
            src_cdf = src_cdf / (src_cdf[-1] if src_cdf[-1] > 0 else 1)
            tgt_cdf = tgt_cdf / (tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1)
            
            # 创建查找表
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # 找到最接近的CDF值
                lookup_table[i] = np.argmin(np.abs(src_cdf[i] - tgt_cdf))
            
            # 应用查找表
            matched[:,:,c] = lookup_table[source[:,:,c].astype(np.uint8)]
        
        matched = matched.astype(np.uint8)
        
    elif mode == 'reinhard':
        # Reinhard色调映射 - 在Lab颜色空间中工作
        # 转换为Lab颜色空间
        source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        
        # 分离Lab通道
        source_l, source_a, source_b = cv2.split(source_lab)
        target_l, target_a, target_b = cv2.split(target_lab)
        
        # 计算均值和标准差
        src_mean_l, src_std_l = np.mean(source_l), np.std(source_l)
        tgt_mean_l, tgt_std_l = np.mean(target_l), np.std(target_l)
        
        src_mean_a, src_std_a = np.mean(source_a), np.std(source_a)
        tgt_mean_a, tgt_std_a = np.mean(target_a), np.std(target_a)
        
        src_mean_b, src_std_b = np.mean(source_b), np.std(source_b)
        tgt_mean_b, tgt_std_b = np.mean(target_b), np.std(target_b)
        
        # 避免除以零
        if src_std_l == 0: src_std_l = 1
        if src_std_a == 0: src_std_a = 1
        if src_std_b == 0: src_std_b = 1
        
        # 标准化并重新缩放
        matched_l = ((source_l - src_mean_l) / src_std_l * tgt_std_l + tgt_mean_l)
        matched_a = ((source_a - src_mean_a) / src_std_a * tgt_std_a + tgt_mean_a)
        matched_b = ((source_b - src_mean_b) / src_std_b * tgt_std_b + tgt_mean_b)
        
        # 合并通道
        matched_lab = cv2.merge([matched_l, matched_a, matched_b])
        
        # 转换回BGR
        matched = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    else:
        return source.astype(np.uint8)
    
    # 如果需要按强度混合
    if strength < 1.0 and orig_source is not None:
        # 确保类型一致
        matched = matched.astype(np.float32)
        orig_source = orig_source.astype(np.float32)
        
        # 线性混合
        blended = cv2.addWeighted(orig_source, 1.0 - strength, matched, strength, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    return matched

def advanced_color_correction(source, target):
    """
    高级颜色校正，专为静态图像设计
    使用多阶段颜色校正和局部颜色保持
    
    Args:
        source: 源图像（模型生成的嘴部区域）
        target: 目标图像（原始图像的嘴部区域）
    
    Returns:
        颜色校正后的源图像
    """
    source = source.copy().astype(np.float32)
    target = target.copy().astype(np.float32)
    
    # 确保图像大小一致
    if source.shape != target.shape:
        target = cv2.resize(target, (source.shape[1], source.shape[0]))
    
    # 步骤1: 全局颜色匹配 - 使用Reinhard算法在Lab空间
    # 转换为Lab颜色空间
    source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
    target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # 分离Lab通道
    source_l, source_a, source_b = cv2.split(source_lab)
    target_l, target_a, target_b = cv2.split(target_lab)
    
    # 计算均值和标准差
    src_mean_l, src_std_l = np.mean(source_l), np.std(source_l)
    tgt_mean_l, tgt_std_l = np.mean(target_l), np.std(target_l)
    
    src_mean_a, src_std_a = np.mean(source_a), np.std(source_a)
    tgt_mean_a, tgt_std_a = np.mean(target_a), np.std(target_a)
    
    src_mean_b, src_std_b = np.mean(source_b), np.std(source_b)
    tgt_mean_b, tgt_std_b = np.mean(target_b), np.std(target_b)
    
    # 避免除以零
    if src_std_l == 0: src_std_l = 1
    if src_std_a == 0: src_std_a = 1
    if src_std_b == 0: src_std_b = 1
    
    # 标准化并重新缩放
    matched_l = ((source_l - src_mean_l) / src_std_l * tgt_std_l + tgt_mean_l)
    matched_a = ((source_a - src_mean_a) / src_std_a * tgt_std_a + tgt_mean_a)
    matched_b = ((source_b - src_mean_b) / src_std_b * tgt_std_b + tgt_mean_b)
    
    # 合并通道
    matched_lab = cv2.merge([matched_l, matched_a, matched_b])
    
    # 转换回BGR
    global_matched = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    # 步骤2: 局部区域细化 - 分割图像为网格进行局部颜色匹配
    h, w = source.shape[:2]
    grid_size = min(h, w) // 4  # 网格大小
    
    # 创建结果图像
    result = global_matched.copy()
    
    # 定义嘴巴区域（通常位于图像下半部分中心）
    mouth_y1 = int(h * 0.5)
    mouth_y2 = int(h * 0.8)
    mouth_x1 = int(w * 0.3)
    mouth_x2 = int(w * 0.7)
    
    # 在嘴巴区域应用直方图匹配
    mouth_src = global_matched[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    mouth_tgt = target.astype(np.uint8)[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    
    for c in range(3):
        # 计算累积直方图
        src_hist, _ = np.histogram(mouth_src[:,:,c].flatten(), 256, [0, 256], density=True)
        tgt_hist, _ = np.histogram(mouth_tgt[:,:,c].flatten(), 256, [0, 256], density=True)
        
        src_cdf = src_hist.cumsum()
        tgt_cdf = tgt_hist.cumsum()
        
        # 归一化
        src_cdf = src_cdf / (src_cdf[-1] if src_cdf[-1] > 0 else 1)
        tgt_cdf = tgt_cdf / (tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1)
        
        # 创建查找表
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # 找到最接近的CDF值
            lookup_table[i] = np.argmin(np.abs(src_cdf[i] - tgt_cdf))
        
        # 应用查找表
        result[mouth_y1:mouth_y2, mouth_x1:mouth_x2, c] = lookup_table[mouth_src[:,:,c]]
    
    # 步骤3: 皮肤色调匹配
    # 皮肤检测 - 使用简单的HSV范围检测
    hsv = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2HSV)
    
    # 宽松的肤色范围
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)
    
    # 创建掩码
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 形态学操作改进掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # 膨胀以覆盖更多区域
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # 将掩码归一化为0-1
    skin_mask_float = skin_mask.astype(np.float32) / 255.0
    
    # 扩展掩码到3通道
    skin_mask_3ch = np.stack([skin_mask_float] * 3, axis=2)
    
    # 从原始目标图像中提取肤色统计信息
    skin_target = target.astype(np.uint8) * skin_mask_3ch
    
    # 只在有肤色的区域计算统计信息
    if np.sum(skin_mask) > 0:
        for c in range(3):
            # 不对黑色区域进行计算
            skin_pixels = skin_target[:,:,c][skin_mask > 0]
            if len(skin_pixels) > 0:
                target_skin_mean = np.mean(skin_pixels)
                
                # 对结果图像的相同区域进行调整
                result_skin = result[:,:,c] * skin_mask_float
                result_skin_mean = np.mean(result_skin[skin_mask > 0]) if np.sum(skin_mask) > 0 else 0
                
                # 如果有有效值，进行调整
                if result_skin_mean > 0:
                    # 计算偏移量
                    offset = target_skin_mean - result_skin_mean
                    
                    # 只对皮肤区域进行调整
                    mask_indices = skin_mask > 0
                    result[mask_indices, c] = np.clip(result[mask_indices, c] + offset, 0, 255)
    
    # 步骤4: 平滑过渡处理 - 皮肤和非皮肤区域的平滑过渡
    # 使用高斯模糊来平滑掩码边缘
    smooth_mask = cv2.GaussianBlur(skin_mask_float, (21, 21), 0)
    smooth_mask_3ch = np.stack([smooth_mask] * 3, axis=2)
    
    # 混合结果图像和全局匹配图像
    # 皮肤区域使用肤色调整结果，非皮肤区域使用全局匹配结果
    final_result = global_matched * (1 - smooth_mask_3ch) + result * smooth_mask_3ch
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def global_color_correction(source_image, target_image, face_coords=None):
    """
    全局颜色校正函数，处理整个图像以消除色差
    
    Args:
        source_image: 源图像（处理后的图像）
        target_image: 目标图像（原始图像）
        face_coords: 面部坐标，用于调整权重 (y1,y2,x1,x2)
        
    Returns:
        全局色彩校正后的图像
    """
    # 深拷贝以避免修改原始图像
    source = source_image.copy()
    target = target_image.copy()
    
    # 如果启用了仅嘴部模式，仅保留嘴部变化，其他区域完全使用原图
    if args.mouth_only and face_coords is not None:
        y1, y2, x1, x2 = face_coords
        h, w = source.shape[:2]
        
        # 嘴部区域通常在面部下半部分，我们仅保留这部分的变化
        mouth_y1 = y1 + int((y2 - y1) * 0.5)  # 面部下半部分
        mouth_x1 = x1 + int((x2 - x1) * 0.25)  # 面部中央区域
        mouth_x2 = x2 - int((x2 - x1) * 0.25)
        
        # 创建一个蒙版，仅保留嘴部区域
        mask = np.zeros((h, w), dtype=np.float32)
        # 嘴部区域设为1，表示使用source图像
        mask[mouth_y1:y2, mouth_x1:mouth_x2] = 1.0
        
        # 平滑过渡
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        # 将蒙版扩展到3通道
        mask = np.stack([mask] * 3, axis=2)
        
        # 混合图像
        result = target * (1 - mask) + source * mask
        return result.astype(np.uint8)
    
    # 正常的全局颜色校正
    # 在Lab色彩空间中进行处理
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    
    # 如果提供了面部坐标，创建面部区域蒙版和权重
    if face_coords is not None:
        y1, y2, x1, x2 = face_coords
        h, w = source.shape[:2]
        
        # 创建权重蒙版，面部区域权重低，其他区域权重高
        weight_mask = np.ones((h, w), dtype=np.float32) * args.color_preserve
        
        # 面部区域的权重降低（使用指定权重）
        face_weight = max(0.0, min(0.5, 1.0 - args.color_preserve))
        weight_mask[y1:y2, x1:x2] = face_weight
        
        # 使用高斯模糊使权重过渡平滑
        weight_mask = cv2.GaussianBlur(weight_mask, (51, 51), 0)
    else:
        # 如果没有面部坐标，使用统一权重
        weight_mask = np.ones(source.shape[:2], dtype=np.float32) * args.color_preserve
    
    # 对每个通道分别应用直方图匹配
    result_lab = source_lab.copy()
    
    # 处理L通道 - 亮度
    source_l = source_lab[:,:,0]
    target_l = target_lab[:,:,0]
    
    # 计算源图像和目标图像的直方图
    src_hist, _ = np.histogram(source_l.flatten(), 256, [0, 256], density=True)
    tgt_hist, _ = np.histogram(target_l.flatten(), 256, [0, 256], density=True)
    
    # 计算累积分布函数
    src_cdf = src_hist.cumsum()
    tgt_cdf = tgt_hist.cumsum()
    
    # 归一化CDF
    src_cdf_normalized = src_cdf / (src_cdf[-1] if src_cdf[-1] > 0 else 1)
    tgt_cdf_normalized = tgt_cdf / (tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1)
    
    # 创建查找表
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = np.argmin(np.abs(src_cdf_normalized[i] - tgt_cdf_normalized))
    
    # 应用查找表
    matched_l = lookup_table[source_l]
    
    # 根据权重混合原始亮度和匹配的亮度
    result_lab[:,:,0] = source_l * (1 - weight_mask) + matched_l * weight_mask
    
    # 处理A通道 - 红绿
    source_a = source_lab[:,:,1]
    target_a = target_lab[:,:,1]
    
    # 直接保留原始A通道
    result_lab[:,:,1] = source_a * (1 - weight_mask) + target_a * weight_mask
    
    # 处理B通道 - 蓝黄
    source_b = source_lab[:,:,2]
    target_b = target_lab[:,:,2]
    
    # 直接保留原始B通道
    result_lab[:,:,2] = source_b * (1 - weight_mask) + target_b * weight_mask
    
    # 将结果从Lab转换回BGR
    result = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    
    # 确保结果在有效范围内
    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        # 处理静态图片输入
        print(f"处理静态图片: {args.face}, 帧率设置为 {args.fps} fps")
        # 使用imread读取图片
        full_frames = [cv2.imread(args.face)]
        if full_frames[0] is None:
            raise ValueError(f"无法读取图片: {args.face}")
        fps = args.fps
        print(f"图片尺寸: {full_frames[0].shape}")

    else:
        # 处理视频输入
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            # 如果无法获取帧率，使用默认值
            print("警告: 无法从视频中获取帧率，使用默认值25 fps")
            fps = 25

        print('从视频中读取帧: {}'.format(args.face))
        print(f"视频帧率: {fps} fps")

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("可用于推理的帧数量: "+str(len(full_frames)))
    print("帧尺寸: {}".format(full_frames[0].shape if full_frames else "无帧"))

    if not args.audio.endswith('.wav'):
        print('提取原始音频...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    
    # 音频预处理 - 增强清晰度
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    # 应用高通滤波器移除低频噪音 - 确保截止频率在有效范围内
    b, a = butter_highpass(80, 16000, order=3)
    wav = filtfilt(b, a, wav)
    
    # 应用低通滤波器移除高频噪音 - 确保截止频率在有效范围内
    b, a = butter_lowpass(5000, 16000, order=3)
    wav = filtfilt(b, a, wav)
    
    # 归一化音频
    wav = wav / np.max(np.abs(wav))
    
    mel = audio.melspectrogram(wav)
    print(f"梅尔谱图形状: {mel.shape}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('梅尔谱图包含NaN值! 使用TTS语音? 在wav文件中添加少量噪声并重试')

    mel_chunks = []
    # 改进的mel分块方法，以更好地同步
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            # 处理最后一块，确保长度正确
            last_chunk = mel[:, len(mel[0]) - mel_step_size:]
            mel_chunks.append(last_chunk)
            break
        current_chunk = mel[:, start_idx : start_idx + mel_step_size]
        mel_chunks.append(current_chunk)
        i += 1

    print(f"梅尔谱图分块数量: {len(mel_chunks)}")

    # 对于静态图像，可能需要复制帧以匹配音频长度
    if args.static:
        print(f"使用静态图片输入: 将单帧复制 {len(mel_chunks)} 次以匹配音频长度")
        if len(full_frames) == 1:
            full_frames = [full_frames[0]] * len(mel_chunks)
    else:
        # 对于视频输入，确保帧数不超过音频长度
        full_frames = full_frames[:len(mel_chunks)]

    # 标准分辨率处理
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    model = load_model(args.checkpoint_path)
    print("模型加载完成，开始推理...")
    
    # 创建视频写入器
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', 
                         cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    print(f"创建视频写入器，尺寸: {frame_w}x{frame_h}，帧率: {fps}")
    
    # 创建用于存储高分辨率面部的临时存储
    hr_faces = {}

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                      total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            print(f"处理批次，形状: img_batch={img_batch.shape}, mel_batch={mel_batch.shape}")
            print(f"使用设备: {device}")
        
        with torch.no_grad():
            frame_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            
            try:
                pred = model(mel_tensor, frame_tensor)
                if i % 10 == 0:
                    print(f"批次 {i+1} 处理成功")
            except Exception as e:
                print(f"处理批次 {i+1} 时出错: {e}")
                print(f"张量形状: 帧={frame_tensor.shape}, 梅尔谱图={mel_tensor.shape}")
                print(f"张量设备: 帧={frame_tensor.device}, 梅尔谱图={mel_tensor.device}")
                raise

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            # 图像模式 - 直接替换
            if args.static and args.image_mode:
                # 保存原始帧的副本用于颜色匹配和比较
                original_frame = f.copy()
                
                # 直接调整大小并替换 - 简单的图像处理模式
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                # 如果启用了颜色匹配，确保与原始图像颜色匹配
                if args.color_match:
                    # 获取原始脸部区域
                    orig_face = f[y1:y2, x1:x2].copy()
                    
                    # 使用高级颜色处理
                    if args.advanced_color:
                        p = advanced_color_correction(p, orig_face)
                    else:
                        # 使用标准颜色匹配算法
                        p = color_match(p, orig_face, mode=args.color_match_mode, strength=args.color_strength)
                
                # 将处理后的面部放回原始图像
                f_copy = f.copy()
                f_copy[y1:y2, x1:x2] = p
                
                # 全局颜色校正（如果启用）
                if args.global_color or args.mouth_only:
                    f = global_color_correction(f_copy, original_frame, face_coords=(y1, y2, x1, x2))
                else:
                    f = f_copy
            else:
                # 标准视频处理模式 - 使用复杂的混合和增强
                # 保存原始帧的副本用于颜色匹配和比较
                original_frame = f.copy()
                
                # 计算面部大小
                face_h, face_w = y2-y1, x2-x1
                
                # 使用高质量的放大算法处理预测的嘴部区域
                p_upscaled = cv2.resize(p.astype(np.uint8), (face_w, face_h), 
                                      interpolation=cv2.INTER_LANCZOS4)
                
                # 获取原始面部区域用于局部颜色匹配
                original_face = original_frame[y1:y2, x1:x2].copy()
                
                # 颜色匹配 - 确保生成的脸部与原始脸部颜色一致
                if (args.color_match or args.video_color_preserve) and not args.static:
                    # 对整个生成的脸部进行颜色匹配，视频处理模式
                    p_upscaled = color_match(p_upscaled.astype(np.uint8), 
                                            original_face.astype(np.uint8), 
                                            mode=args.color_match_mode,
                                            strength=args.color_strength)
                elif args.color_match and args.static:
                    # 静态图像处理模式
                    p_upscaled = color_match(p_upscaled.astype(np.uint8), 
                                            original_face.astype(np.uint8), 
                                            mode=args.color_match_mode,
                                            strength=args.color_strength)
                
                # 确保两个面部数组类型一致以避免类型不匹配错误
                original_face = original_face.astype(np.float32)
                p_upscaled = p_upscaled.astype(np.float32)
                
                # 只处理下半部分面部，上半部分保持不变
                mouth_y = int(face_h * 0.45)  # 调整嘴巴开始的位置，避免影响到下巴
                
                # 创建HSV版本，用于颜色空间处理和牙齿检测
                original_face_hsv = cv2.cvtColor(original_face.astype(np.uint8), cv2.COLOR_BGR2HSV)
                proc_face_hsv = cv2.cvtColor(p_upscaled.astype(np.uint8), cv2.COLOR_BGR2HSV)
                
                # 提取亮度通道，用于后续牙齿检测
                _, _, orig_v = cv2.split(original_face_hsv)
                _, _, proc_v = cv2.split(proc_face_hsv)
                
                # 使用全新的自适应多层混合蒙版，完全消除几何边界
                mask = np.zeros((face_h, face_w), dtype=np.float32)
                
                # 计算嘴巴和面部中心位置
                mouth_center_y = int(mouth_y + face_h * 0.10)  # 嘴巴中心位置
                center_x = face_w // 2  # 水平中心
                
                # 创建索引网格
                y_indices, x_indices = np.mgrid[:face_h, :face_w]
                
                # 首先创建一个非常有限的嘴部中心区域
                mouth_radius = min(face_w, face_h) * 0.2  # 更小的嘴部区域半径
                
                # 计算到嘴部中心的距离
                dist_from_mouth = np.sqrt(((y_indices - mouth_center_y) / (face_h * 0.4))**2 + 
                                        ((x_indices - center_x) / (face_w * 0.5))**2)
                
                # 基础混合系数 - 整体降低强度
                mouth_intensity = 0.25
                
                # 创建逐渐衰减的蒙版
                # 核心区域 - 使用非线性函数实现更自然的过渡
                mask = mouth_intensity * np.exp(-3.0 * dist_from_mouth)
                
                # 确保上半部分面部不受影响 - 使用平滑过渡而非硬边界
                upper_fade = np.clip((y_indices - (mouth_y - 20)) / 40.0, 0, 1)
                mask = mask * upper_fade
                
                # 确保下巴区域快速衰减
                lower_boundary = mouth_center_y + face_h * 0.15
                lower_fade = np.clip(1.0 - (y_indices - lower_boundary) / (face_h * 0.2), 0, 1)
                mask = mask * lower_fade
                
                # 移除边缘区域的任何影响
                edge_margin = int(min(face_w, face_h) * 0.1)
                edge_x_fade_left = np.clip(x_indices / edge_margin, 0, 1)
                edge_x_fade_right = np.clip((face_w - x_indices) / edge_margin, 0, 1)
                edge_y_fade_bottom = np.clip((face_h - y_indices) / edge_margin, 0, 1)
                
                # 将边缘衰减应用到蒙版
                mask = mask * edge_x_fade_left * edge_x_fade_right * edge_y_fade_bottom
                
                # 应用非线性调整，以创建更自然的渐变
                mask = np.power(mask, 1.5)  # 指数调整使边缘更加柔和
                
                # 最终平滑，确保没有任何硬边界
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                
                # 锐化处理，但保持颜色平衡
                mouth_region = p_upscaled.copy()
                
                # 使用更轻微的锐化
                kernel_strength = max(0.6, min(0.8, args.mouth_enhance))
                kernel = np.array([[-0.1,-0.1,-0.1], [-0.1,2.0,-0.1], [-0.1,-0.1,-0.1]]) * kernel_strength
                kernel[1][1] = kernel[1][1] * (1/kernel_strength) + 0.2
                
                # 只在嘴部中心区域应用锐化，完全避开边缘
                mouth_hsv = cv2.cvtColor(mouth_region.astype(np.uint8), cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(mouth_hsv)
                
                # 只处理强烈混合的区域，避免在边缘创建伪影
                core_region = mask > 0.1
                
                # 创建原始嘴部区域的副本
                v_enhanced = v.copy()
                
                # 只在核心区域应用锐化
                if np.any(core_region):
                    # 创建一个只包含核心区域的临时数组
                    v_core = v[core_region]
                    # 应用锐化
                    v_core_enhanced = cv2.filter2D(v_core, -1, kernel)
                    # 将增强后的值放回原始数组
                    v_enhanced[core_region] = v_core_enhanced
                
                # 用增强版本替换原来的亮度通道
                v = v_enhanced
                
                # 降低饱和度，保持更自然的颜色
                s = np.clip(s * 0.98, 0, 255).astype(np.uint8)
                
                # 重新合并通道
                mouth_hsv = cv2.merge([h, s, v])
                mouth_region = cv2.cvtColor(mouth_hsv, cv2.COLOR_HSV2BGR)
                
                # 牙齿处理 - 只处理高概率区域
                # 使用更保守的牙齿检测
                teeth_threshold = 200  # 非常高的阈值，确保只有最亮的区域被识别为牙齿
                
                # 只检测嘴部中心区域的牙齿
                mouth_center_region = dist_from_mouth < 0.5
                teeth_mask = np.zeros_like(orig_v)
                teeth_mask[mouth_center_region] = (orig_v[mouth_center_region] > teeth_threshold).astype(np.uint8) * 255
                
                # 形态学操作清理牙齿区域
                kernel = np.ones((2, 2), np.uint8)
                teeth_mask = cv2.erode(teeth_mask, kernel, iterations=1)
                teeth_mask = cv2.dilate(teeth_mask, kernel, iterations=1)
                
                # 平滑牙齿边缘
                teeth_mask = cv2.GaussianBlur(teeth_mask, (3, 3), 0)
                
                # 将蒙版归一化
                teeth_mask_norm = teeth_mask.astype(np.float32) / 255.0
                
                # 对每个颜色通道单独处理
                for c in range(3):
                    # 计算原始面部区域的颜色统计信息
                    orig_c = original_face[:, :, c]
                    orig_mean = np.mean(orig_c)
                    orig_std = np.std(orig_c)
                    
                    # 计算处理后区域的颜色统计信息
                    proc_c = mouth_region[:, :, c]
                    proc_mean = np.mean(proc_c)
                    proc_std = np.std(proc_c)
                    
                    # 避免除以零
                    if proc_std == 0:
                        proc_std = 1
                    
                    # 创建全局颜色转换函数，确保颜色整体匹配
                    normalized = (proc_c.astype(np.float32) - proc_mean) / proc_std
                    matched = normalized * orig_std * 0.8 + orig_mean
                    
                    # 只对核心区域应用颜色匹配，避免边缘处的不自然转换
                    matched_weighted = proc_c.astype(np.float32) * (1.0 - core_region.astype(np.float32)) + \
                                      matched * core_region.astype(np.float32)
                    
                    # 牙齿区域特殊处理 - 柔和增亮
                    teeth_color = orig_mean + 10  # 比原始颜色略亮一点
                    teeth_blend_factor = 0.3  # 非常轻的混合
                    
                    # 应用牙齿增强
                    matched_weighted = matched_weighted * (1.0 - teeth_mask_norm) + \
                                     (teeth_color * teeth_blend_factor + matched_weighted * (1 - teeth_blend_factor)) * teeth_mask_norm
                    
                    # 更新处理后的通道
                    mouth_region[:, :, c] = np.clip(matched_weighted, 0, 255).astype(np.uint8)
                
                # 扩展蒙版到3通道
                mask3d = np.expand_dims(mask, axis=2)
                mask3d = np.repeat(mask3d, 3, axis=2)
                
                # 混合原始图像和处理后的图像 - 确保类型一致
                mouth_region = mouth_region.astype(np.float32)
                original_face = original_face.astype(np.float32)
                
                # 使用cv2.addWeighted确保类型匹配
                blended_face = cv2.addWeighted(original_face, 1.0 - mask3d.max(), mouth_region, mask3d.max(), 0)
                
                # 确保输出类型正确
                blended_face = np.clip(blended_face, 0, 255).astype(np.uint8)
                
                # 将混合后的面部放回原始图像
                f_copy = f.copy()
                f_copy[y1:y2, x1:x2] = blended_face
                
                # 全局颜色校正（如果启用）
                if args.global_color or args.mouth_only:
                    f = global_color_correction(f_copy, original_face, face_coords=(y1, y2, x1, x2))
                else:
                    f = f_copy
            
            out.write(f)
    
    out.release()
    
    # 检查结果是否存在
    if not os.path.exists('temp/result.avi') or os.path.getsize('temp/result.avi') == 0:
        raise RuntimeError('视频生成失败，输出文件为空或不存在')

    # 增强最终输出视频质量
    temp_avi = 'temp/result.avi'
    final_output = args.outfile
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(final_output)), exist_ok=True)
    
    # 超分辨率增强输出视频
    print("增强最终视频质量...")
    # 首先生成不带音频的增强版本，特别增强边缘锐度和细节
    # 如果启用了色彩匹配，使用更温和的增强参数，保持颜色一致性
    if args.color_match or args.video_color_preserve or args.advanced_color:
        # 更保守的视频处理参数，保持颜色
        enhance_cmd = 'ffmpeg -y -i {} -vf "scale=iw*1.5:ih*1.5:flags=lanczos, eq=brightness=0:saturation=1.0:contrast=1.0, unsharp=3:3:0.8:3:3:0.0" -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p temp/enhanced.mp4'.format(temp_avi)
    else:
        # 标准增强参数
        enhance_cmd = 'ffmpeg -y -i {} -vf "scale=iw*1.5:ih*1.5:flags=lanczos, eq=brightness=0.05:saturation=1.2:contrast=1.1, unsharp=5:5:1.5:5:5:0.0, unsharp=3:3:1.0:3:3:0.0" -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p temp/enhanced.mp4'.format(temp_avi)
    
    try:
        subprocess.call(enhance_cmd, shell=True)
        
        # 检查输出文件
        if not os.path.exists('temp/enhanced.mp4') or os.path.getsize('temp/enhanced.mp4') == 0:
            print("警告: 增强过程失败，使用原始视频")
            enhanced_path = temp_avi
        else:
            enhanced_path = 'temp/enhanced.mp4'
            
        # 添加音频到最终输出
        audio_cmd = 'ffmpeg -y -i {} -i {} -c:v copy -c:a aac -strict experimental -shortest {}'.format(
            enhanced_path, args.audio, final_output) 
        
        subprocess.call(audio_cmd, shell=True)
        
        if not os.path.exists(final_output):
            raise RuntimeError(f"最终视频 {final_output} 创建失败")
            
        print('增强后的视频已保存至 {}'.format(final_output))
    
    except Exception as e:
        print(f"视频处理过程中发生错误: {e}")
        print("尝试使用备用方法...")
        
        # 简单的备用方法 - 直接复制视频并添加音频
        direct_cmd = 'ffmpeg -y -i {} -i {} -c:v copy -c:a aac -strict experimental -shortest {}'.format(
            temp_avi, args.audio, final_output)
        subprocess.call(direct_cmd, shell=True)
        
        if os.path.exists(final_output):
            print(f"成功使用备用方法创建视频: {final_output}")
        else:
            print(f"所有视频处理方法失败，请检查FFmpeg安装和输入文件")
    
    # 删除临时文件
    if not args.audio.endswith('.wav'):
        try:
            os.remove('temp/temp.wav')
        except:
            pass

if __name__ == '__main__':
	main()
