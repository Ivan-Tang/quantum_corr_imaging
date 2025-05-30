import cv2
import numpy as np
import os
from tqdm import tqdm


def batch_image_fusion(folder_path, start=1, end=100):
    """
    批量图像平均融合
    :return: (融合后的图像, 有效图像数)
    """
    accumulator = None
    valid_count = 0
    base_size = None

    with tqdm(total=end - start + 1, desc="Processing Images") as pbar:
        for i in range(start, end + 1):
            img_path = os.path.join(folder_path, f"signal/{i}.jpg")

            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("OpenCV读取失败")

                if base_size is None:
                    base_size = (img.shape[1], img.shape[0])
                    accumulator = np.zeros_like(img, dtype=np.float64)  # 改为float64提高精度
                else:
                    img = cv2.resize(img, base_size)

                accumulator += img.astype(np.float64)
                valid_count += 1

            except Exception as e:
                print(f"\n跳过错误文件 {os.path.basename(img_path)}: {str(e)}")

            pbar.update(1)

    if valid_count == 0:
        raise RuntimeError("没有找到有效图像")

    avg_img = np.clip(accumulator , 0, 255).astype(np.uint8)
    return avg_img, valid_count  # 返回两个值


if __name__ == "__main__":
    try:
        image_folder = r"E:\1"
        output_path = r"E:\1\fused.jpg"

        # 接收两个返回值
        fused_img, count = batch_image_fusion(image_folder)

        cv2.imwrite(output_path, fused_img)
        print(f"\n成功融合 {count} 张图像，结果保存至: {output_path}")

        cv2.imshow('Fused Result', fused_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"处理失败: {str(e)}")
