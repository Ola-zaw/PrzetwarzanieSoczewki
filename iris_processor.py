import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class IrisProcessor:
    """Klasa logiczna zawierająca algorytmy z instrukcji."""
    
    @staticmethod
    def to_grayscale(img):
        if len(img.shape) == 3:
            return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return img

    @staticmethod
    def calculate_base_threshold(gray_img):
        h, w = gray_img.shape
        return np.sum(gray_img) / (h * w)

    @staticmethod
    def binarize(gray_img, threshold):
        return (gray_img < threshold).astype(np.uint8) * 255
    
    @staticmethod
    def filter_min_max(img, size, mode='min'):
        pad_w = size // 2
        pad_img = np.pad(img, pad_w, mode='edge')
        windows = sliding_window_view(pad_img, window_shape=(size, size))
        
        if mode == 'min':
            return np.min(windows, axis=(-2, -1)).astype(np.uint8)
        else:
            return np.max(windows, axis=(-2, -1)).astype(np.uint8)
        
    @staticmethod
    def apply_morphology(img, operation, size):
        if "Brak" in operation:
            return img
        elif "Usuń rzęsy" in operation:
            tmp = IrisProcessor.filter_min_max(img, size, 'max')
            return IrisProcessor.filter_min_max(tmp, size, 'min')
        elif "Zalej refleksy" in operation:
            tmp = IrisProcessor.filter_min_max(img, size, 'min')
            return IrisProcessor.filter_min_max(tmp, size, 'max')
        elif "Tylko powiększ czarne" in operation:
            return IrisProcessor.filter_min_max(img, size, 'min')
        elif "Tylko powiększ białe" in operation:
            return IrisProcessor.filter_min_max(img, size, 'max')
        return img
    
    @staticmethod
    def find_center_via_projections(binary_img):
        inverted = np.where(binary_img == 0, 1, 0)
        proj_y = np.sum(inverted, axis=1)
        max_y = np.max(proj_y)
        y_indices = np.where(proj_y == max_y)[0]
        center_y = int(np.mean(y_indices)) 
        
        proj_x = np.sum(inverted, axis=0)
        max_x = np.max(proj_x)
        x_indices = np.where(proj_x == max_x)[0]
        center_x = int(np.mean(x_indices))
        
        return center_x, center_y

    @staticmethod
    def draw_crosshair(img, x, y, size=20, color=(255, 0, 0)):
        if len(img.shape) == 2:
            img_color = np.stack([img, img, img], axis=-1)
        else:
            img_color = img.copy()
            
        h, w = img_color.shape[:2]
        
        x_start = max(0, x - size)
        x_end = min(w, x + size)
        img_color[y, x_start:x_end] = color
        
        y_start = max(0, y - size)
        y_end = min(h, y + size)
        img_color[y_start:y_end, x] = color
        
        return img_color