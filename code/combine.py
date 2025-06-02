import cv2
import numpy as np
import os
from tqdm import tqdm
import time



def batch_image_fusion(base_path, fusion_num=None):
    """
    批量图像平均融合
    base_path: 图像文件夹路径
    fusion_num: 需要融合的图像数量
    return: 融合后的图像
    """

    start_time = time.time()
    object_dirs = []
    for name in sorted(os.listdir(base_path)):
        obj_dir = os.path.join(base_path, name)
        if os.path.isdir(obj_dir):
            object_dirs.append(obj_dir)
    
    results_img = []
    for obj_dir in object_dirs:
        obj_dir = os.path.join(obj_dir, 'signal')
        accumulator = None
        base_size = None 
        img_paths = sorted([
                os.path.join(obj_dir, f) for f in os.listdir(obj_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        if fusion_num is None:
            fusion_num = len(img_paths) #如果没有fusion_num则用全部图片叠加

        if len(img_paths) < fusion_num:
            raise ValueError(f"图像数量 {len(img_paths)} 小于融合数量 {fusion_num}")
        
        for i in range(fusion_num):
            img = cv2.imread(img_paths[i])
            if base_size is None:
                base_size = (img.shape[1], img.shape[0])
                accumulator = np.zeros_like(img, dtype=np.float64)  # 改为float64提高精度

            accumulator += img.astype(np.float64)
        
        accumulator /= (fusion_num / 20) 
        results_img.append(np.clip(accumulator, 0, 255).astype(np.uint8))
        print(f"融合完成 {obj_dir}")

    end_time = time.time()
    print(f"总耗时 {end_time - start_time:.2f}s")
    avg_time = (end_time - start_time) / len(object_dirs)
    print(f"平均耗时 {avg_time:.2f}s/个体")
    
    return results_img


if __name__ == "__main__":
    img_path = 'data/test/'
    save_path = 'reports/imgs/'
    os.makedirs(save_path, exist_ok=True)
    fusion_num = 100
    results = batch_image_fusion(img_path, fusion_num)
    print(len(results))
    for i, img in enumerate(results):
        filename = f'fused_image_{i}.png'
        full_path = os.path.join(save_path, filename)
        cv2.imwrite(full_path, img)
        print(f"保存完成 {full_path}")

