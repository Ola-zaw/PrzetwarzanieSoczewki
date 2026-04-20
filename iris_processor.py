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
    
    # @staticmethod
    # def find_center_via_projections(binary_img):
    #     inverted = np.where(binary_img == 0, 1, 0)
    #     proj_y = np.sum(inverted, axis=1)
    #     max_y = np.max(proj_y)
    #     y_indices = np.where(proj_y == max_y)[0]
    #     center_y = int(np.mean(y_indices)) 
        
    #     proj_x = np.sum(inverted, axis=0)
    #     max_x = np.max(proj_x)
    #     x_indices = np.where(proj_x == max_x)[0]
    #     center_x = int(np.mean(x_indices))
        
    #     return center_x, center_y

    # @staticmethod
    # def draw_crosshair(img, x, y, size=20, color=(255, 0, 0)):
    #     if len(img.shape) == 2:
    #         img_color = np.stack([img, img, img], axis=-1)
    #     else:
    #         img_color = img.copy()
            
    #     h, w = img_color.shape[:2]
        
    #     x_start = max(0, x - size)
    #     x_end = min(w, x + size)
    #     img_color[y, x_start:x_end] = color
        
    #     y_start = max(0, y - size)
    #     y_end = min(h, y + size)
    #     img_color[y_start:y_end, x] = color
        
    #     return img_color


    @staticmethod
    def find_center_and_radius_via_projections(binary_img):
        THRESHOLD_RATIO = 0.9
        
        inverted = np.where(binary_img == 0, 1, 0)
        
        proj_y = np.sum(inverted, axis=1)
        proj_x = np.sum(inverted, axis=0)
        
        def get_bounds(proj):
            max_val = np.max(proj)
            if max_val == 0:
                return 0, 0
            
            threshold = max_val * THRESHOLD_RATIO * 0.2
            valid_indices = np.where(proj > threshold)[0]
            if len(valid_indices) == 0:
                return 0, 0
                
            return valid_indices[0], valid_indices[-1]
            
        top, bottom = get_bounds(proj_y)
        left, right = get_bounds(proj_x)
        
        radius_x = (right - left) / 2
        radius_y = (bottom - top) / 2
        radius = int((radius_x + radius_y) / 2)
        
        max_y = np.max(proj_y)
        y_indices = np.where((proj_y <= max_y) & (proj_y > max_y * THRESHOLD_RATIO))[0]
        center_y = int(np.mean(y_indices)) 
        
        proj_x = np.sum(inverted, axis=0)
        max_x = np.max(proj_x)
        x_indices = np.where((proj_x <= max_x) & (proj_x > max_x * THRESHOLD_RATIO))[0]
        center_x = int(np.mean(x_indices))

        return center_x, center_y, radius


    @staticmethod
    def draw_crosshair_and_circle(img, x, y, r, cross_size=20, color=(255, 0, 0)):
        if len(img.shape) == 2:
            img_color = np.stack([img, img, img], axis=-1)
        else:
            img_color = img.copy()
            
        h, w = img_color.shape[:2]
        
        x_start = max(0, x - cross_size)
        x_end = min(w, x + cross_size)
        img_color[y, x_start:x_end] = color
        
        y_start = max(0, y - cross_size)
        y_end = min(h, y + cross_size)
        img_color[y_start:y_end, x] = color
        
        y_idx, x_idx = np.ogrid[:h, :w]
        
        dist_from_center_sq = (x_idx - x)**2 + (y_idx - y)**2
        
        ring_thickness = 2
        ring_mask = (dist_from_center_sq <= r**2) & (dist_from_center_sq >= (max(0, r - ring_thickness))**2)
        
        img_color[ring_mask] = color
        
        return img_color
    


    @staticmethod
    def find_iris_radius(gray_img, cx, cy, pupil_radius):
        """
        Wyznacza promień tęczówki analizując wyłącznie poziomy pas przechodzący przez środek źrenicy.
        Ignoruje zakłócenia od powiek i rzęs z góry i z dołu.
        """
        h, w = gray_img.shape
        
        # Wycinamy poziomy pasek o wysokości 20 pikseli wokół środka źrenicy
        strip_height = 10 
        y_start = max(0, cy - strip_height)
        y_end = min(h, cy + strip_height)
        
        # Pobieramy pasek i uśredniamy go pionowo, aby uzyskać jeden stabilny ciąg wartości (1D)
        horizontal_strip = gray_img[y_start:y_end, :]
        profile = np.mean(horizontal_strip, axis=0)
        
        # Obliczamy gradient (pochodną) - czyli różnicę jasności między sąsiednimi pikselami.
        # W miejscu przejścia ciemnej tęczówki w jasną twardówkę będzie skok.
        gradient = np.abs(np.diff(profile))
        
        ignore_margin = int(pupil_radius * 1.2)
        safe_left = max(0, cx - ignore_margin)
        safe_right = min(w - 1, cx + ignore_margin)
        gradient[safe_left:safe_right] = 0
        
        # największy skok po lewej stronie
        left_half = gradient[:cx]
        left_edge_x = np.argmax(left_half) if len(left_half) > 0 else 0
        
        # największy skok po prawej stronie
        right_half = gradient[cx:]
        right_edge_x = cx + np.argmax(right_half) if len(right_half) > 0 else 0
        
        r_left = cx - left_edge_x
        r_right = right_edge_x - cx
        
        # uśredniamy wynik
        iris_radius = int((r_left + r_right) / 2)
        
        # zabezpieczenie przed błędem: tęczówka nie może być poza zdjęciem
        if iris_radius < pupil_radius:
            iris_radius = pupil_radius + 20 # Wartość domyślna awaryjna
            
        return iris_radius
    


    @staticmethod
    def unwrap_iris(image, cx, cy, r_pupil, r_iris, width=360, height=60):
        """
        Przekształca pierścień tęczówki w prostokąt (normalizacja).
        width: rozdzielczość kątowa (0-360 stopni)
        height: rozdzielczość radialna (odległość między źrenicą a krawędzią tęczówki)
        """
        # Przygotowujemy pusty obraz wynikowy
        if len(image.shape) == 3:
            unwrapped = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            unwrapped = np.zeros((height, width), dtype=np.uint8)

        # Tworzymy siatki kątów (theta) i promieni (rho)
        # theta: od 0 do 2*pi
        # rho: od 0 (źrenica) do 1 (krawędź tęczówki)
        thetas = np.linspace(0, 2 * np.pi, width)
        rhos = np.linspace(0, 1, height)

        # Tworzymy macierze współrzędnych dla całego prostokąta naraz
        theta_grid, rho_grid = np.meshgrid(thetas, rhos)

        # Wyznaczamy promień dla każdego punktu w prostokącie
        r_grid = r_pupil + rho_grid * (r_iris - r_pupil)

        # Konwersja na współrzędne kartezjańskie (x, y) na oryginalnym obrazie
        x_grid = cx + r_grid * np.cos(theta_grid)
        y_grid = cy + r_grid * np.sin(theta_grid)

        # Rzutujemy współrzędne na macierz obrazu (zaokrąglanie i obcinanie do granic)
        x_grid = np.clip(np.round(x_grid), 0, image.shape[1] - 1).astype(int)
        y_grid = np.clip(np.round(y_grid), 0, image.shape[0] - 1).astype(int)

        # Pobieramy wartości pikseli z oryginalnego obrazu
        unwrapped = image[y_grid, x_grid]
        
        return unwrapped