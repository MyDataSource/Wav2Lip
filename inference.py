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
					help='Factor to enhance mouth region sharpness (1.0 = normal, 1.5 = sharper, 0.5 = smoother)')

args = parser.parse_args()
args.img_size = 96  # 固定为96，原始模型仅支持这个尺寸
print(f"Using fixed image size of {args.img_size}x{args.img_size} for face crops (model limitation)")

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

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
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

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

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
print('CUDA available: {}, MPS available: {}, MPS built: {}'.format(
    torch.cuda.is_available(), 
    torch.backends.mps.is_available(),
    torch.backends.mps.is_built()
))

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

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames from: {}'.format(args.face))

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: "+str(len(full_frames)))
    print("Frame dimensions: {}".format(full_frames[0].shape if full_frames else "No frames"))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
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
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

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

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    # 标准分辨率处理
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    model = load_model(args.checkpoint_path)
    print("Model loaded, starting inference...")
    
    # 创建视频写入器
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', 
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    print(f"Created video writer with dimensions: {frame_w}x{frame_h}, fps: {fps}")
    
    # 创建一个4倍大小的临时存储，用于超分辨率处理
    hr_faces = {}  # 用于存储高分辨率版本的面部

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                        total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            print(f"Processing batch with shapes: img_batch={img_batch.shape}, mel_batch={mel_batch.shape}")
            print(f"Using model on {device} device")
        
        with torch.no_grad():
            frame_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            
            try:
                pred = model(mel_tensor, frame_tensor)
                if i % 10 == 0:
                    print(f"Batch {i+1} processed successfully")
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                print(f"Frame tensor shape: {frame_tensor.shape}, Mel tensor shape: {mel_tensor.shape}")
                print(f"Frame tensor device: {frame_tensor.device}, Mel tensor device: {mel_tensor.device}")
                raise

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            # 计算面部大小
            face_h, face_w = y2-y1, x2-x1
            
            # 使用高质量的放大算法处理预测的嘴部区域
            p_upscaled = cv2.resize(p.astype(np.uint8), (face_w, face_h), 
                                   interpolation=cv2.INTER_LANCZOS4)
            
            # 创建面部区域的原始图像
            original_face = f[y1:y2, x1:x2].copy()
            
            # 只处理下半部分面部，上半部分保持不变
            mouth_y = int(face_h * 0.45)  # 调整嘴巴开始的位置，避免影响到下巴
            
            # 创建HSV版本，用于颜色空间处理和牙齿检测
            original_face_hsv = cv2.cvtColor(original_face.astype(np.uint8), cv2.COLOR_BGR2HSV)
            proc_face_hsv = cv2.cvtColor(p_upscaled, cv2.COLOR_BGR2HSV)
            
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
            mouth_hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
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
            
            # 混合原始图像和处理后的图像
            mouth_region = mouth_region.astype(np.float32)
            original_face = original_face.astype(np.float32)
            
            # 使用矢量化操作混合图像
            blended_face = (1.0 - mask3d) * original_face + mask3d * mouth_region
            
            # 确保输出类型正确
            blended_face = np.clip(blended_face, 0, 255).astype(np.uint8)
            
            # 将混合后的面部放回原始图像
            f[y1:y2, x1:x2] = blended_face
            
            out.write(f)
    
    out.release()

    # 增强最终输出视频质量
    temp_avi = 'temp/result.avi'
    final_output = args.outfile
    
    # 超分辨率增强输出视频
    print("Enhancing final video quality...")
    # 首先生成不带音频的增强版本，特别增强边缘锐度和细节
    enhance_cmd = 'ffmpeg -y -i {} -vf "scale=iw*1.5:ih*1.5:flags=lanczos, eq=brightness=0.05:saturation=1.2:contrast=1.1, unsharp=5:5:1.5:5:5:0.0, unsharp=3:3:1.0:3:3:0.0" -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p temp/enhanced.mp4'.format(temp_avi)
    subprocess.call(enhance_cmd, shell=True)
    
    # 然后添加音频
    audio_cmd = 'ffmpeg -y -i temp/enhanced.mp4 -i {} -c:v copy -c:a aac -strict experimental -shortest {}'.format(args.audio, final_output) 
    subprocess.call(audio_cmd, shell=True)
    
    print('Enhanced video saved to {}'.format(final_output))
    
    # 删除临时文件
    if not args.audio.endswith('.wav'):
        os.remove('temp/temp.wav')

if __name__ == '__main__':
	main()
