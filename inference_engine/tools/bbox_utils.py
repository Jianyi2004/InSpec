import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# ===== 中文文本绘制函数 =====
def draw_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), bg_color=(255, 0, 0)):
    """
    在图片上绘制中文文本
    
    Args:
        img: numpy数组格式的图片
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文字颜色 (R, G, B)
        bg_color: 背景颜色 (R, G, B)
    
    Returns:
        绘制文本后的图片
    """
    # 转换为PIL图片
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试加载中文字体
    try:
        # 尝试常见的中文字体路径
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux中文字体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # Linux中文字体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:/Windows/Fonts/simhei.ttf",      # Windows
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 获取文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x, y = position
    
    # 绘制背景矩形
    draw.rectangle([x, y - text_height - 5, x + text_width + 10, y + 5], fill=bg_color)
    
    # 绘制文本
    draw.text((x + 5, y - text_height), text, font=font, fill=color)
    
    # 转换回numpy数组
    return np.array(pil_img)

# ===== 统一边界框可视化函数 =====
def draw_bounding_boxes_with_cache(img_path, json_path,draw_text = False, save_to_cache=True):
    """
    在图片上绘制边界框，支持缓存到临时文件夹
    
    Args:
        img_path: 图片路径
        json_path: JSON标注文件路径
        save_to_cache: 是否保存到缓存文件夹
    
    Returns:
        如果save_to_cache=True: 返回缓存图片的路径
        如果save_to_cache=False: 返回带有边界框的图片数组 (numpy array)
    """
    # 读取图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 如果JSON文件存在，读取标注信息
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            
            # 绘制每个标注框
            for shape in ann.get("shapes", []):
                if shape.get("shape_type") == "rectangle":
                    points = shape.get("points", [])
                    if len(points) >= 2:
                        # 获取矩形的两个对角点
                        x1, y1 = int(points[0][0]), int(points[0][1])
                        x2, y2 = int(points[1][0]), int(points[1][1])
                        
                        # 确保坐标顺序正确
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        
                        # 绘制矩形框
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # 红色框，线宽3
                        
                        # 添加标签文本
                        label = shape.get("label", "")
                        
                        if label and draw_text:
                            # 计算文本位置
                            text_x = x_min
                            text_y = y_min - 10 if y_min > 30 else y_min + 25
                            
                            # 使用中文文本绘制函数
                            img = draw_chinese_text(img, label, (text_x, text_y), 
                                                  font_size=20, color=(255, 255, 255), bg_color=(255, 0, 0))
                
                elif shape.get("shape_type") == "polygon":
                    # 处理多边形标注
                    points = shape.get("points", [])
                    if len(points) >= 3:
                        # 转换为numpy数组
                        pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # 绘制多边形
                        cv2.polylines(img, [pts], True, (0, 255, 0), 3)  # 绿色框，线宽3
                        
                        # 添加标签
                        label = shape.get("label", "")
                        if label and len(points) > 0:
                            text_x, text_y = int(points[0][0]), int(points[0][1])
                            # 使用中文文本绘制函数
                            img = draw_chinese_text(img, label, (text_x, text_y - 10), 
                                                  font_size=20, color=(255, 255, 255), bg_color=(0, 255, 0))
        
        except Exception as e:
            if 'st' in globals():
                st.warning(f"⚠️ 读取标注文件失败: {e}")
            else:
                print(f"⚠️ 读取标注文件失败: {e}")
    
    if save_to_cache:
        # 保存到缓存文件夹
        cache_dir = "/data_all/share/IndustrialDefectDetection"
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名（基于原图片路径的hash）
        import hashlib
        img_hash = hashlib.md5(img_path.encode('utf-8')).hexdigest()
        cache_filename = f"boxed_{img_hash}_{os.path.basename(img_path)}"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # 保存图片
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', img_bgr)[1].tofile(cache_path)
        
        return cache_path
    else:
        return img