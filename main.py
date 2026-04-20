import sys
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QComboBox)
from PyQt5.QtCore import Qt

from przegladarka_obrazow import PrzegladarkaObrazow
from projekcje import ProjekcjaGorna, ProjekcjaBoczna
from iris_processor import IrisProcessor
from iris_worker import IrisWorker

class IrisMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentacja tęczówki")
        self.current_step = 0
        self.original_image = None
        self.current_processed_image = None
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # --- Pasek ładowania ---
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("Wczytaj obraz")
        self.btn_load.clicked.connect(self.load_image)
        top_layout.addWidget(self.btn_load)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # --- Przeglądarka i Projekcje ---
        self.grid = QGridLayout()
        self.grid.setSpacing(0)

        self.proj_gora = ProjekcjaGorna()
        self.proj_boczna = ProjekcjaBoczna()
        self.viewer = PrzegladarkaObrazow()
        self.viewer.visible_rect_changed.connect(self.update_projections_from_rect)
        self.viewer.setMinimumSize(600, 400)

        self.grid.addWidget(self.proj_gora, 0, 1)   
        self.grid.addWidget(self.proj_boczna, 1, 0) 
        self.grid.addWidget(self.viewer, 1, 1)

        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(1, 1)
        
        self.proj_gora.setVisible(False)
        self.proj_boczna.setVisible(False)
        layout.addLayout(self.grid)

        # --- Suwak Źrenicy ---
        self.control_layout = QHBoxLayout()
        self.lbl_param = QLabel("Parametr x_I (Źrenica):")
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(1, 50)
        self.slider_x.setValue(10)
        self.slider_x.setTickPosition(QSlider.TicksBelow)
        self.lbl_slider_val = QLabel("1.0")
        
        self.control_layout.addWidget(self.lbl_param)
        self.control_layout.addWidget(self.slider_x)
        self.control_layout.addWidget(self.lbl_slider_val)
        layout.addLayout(self.control_layout)
        self.slider_x.valueChanged.connect(self.on_slider_changed)

        # --- Suwak Tęczówki ---
        self.control_layout_iris = QHBoxLayout()
        self.lbl_param_iris = QLabel("Parametr x_P (Tęczówka):")
        self.slider_x_iris = QSlider(Qt.Horizontal)
        self.slider_x_iris.setRange(1, 100)
        self.slider_x_iris.setValue(10)
        self.slider_x_iris.setTickPosition(QSlider.TicksBelow)
        self.lbl_slider_val_iris = QLabel("1.0")
        
        self.control_layout_iris.addWidget(self.lbl_param_iris)
        self.control_layout_iris.addWidget(self.slider_x_iris)
        self.control_layout_iris.addWidget(self.lbl_slider_val_iris)
        layout.addLayout(self.control_layout_iris)
        self.slider_x_iris.valueChanged.connect(self.on_slider_iris_changed)
        
        self.set_controls_visible(visible_pupil=False, visible_iris=False)

        # --- Morfologia ---
        self.morph_widget = QWidget()
        self.morph_layout = QVBoxLayout(self.morph_widget)
        opcje = ["Brak", "Usuń rzęsy (Max -> Min)", "Zalej refleksy (Min -> Max)", "Tylko powiększ czarne (Min)", "Tylko powiększ białe (Max)"]
        
        row1 = QHBoxLayout()
        self.combo_morph_1 = QComboBox()
        self.combo_morph_1.addItems(opcje)
        self.slider_morph_1 = QSlider(Qt.Horizontal)
        self.slider_morph_1.setRange(1, 15)
        self.lbl_morph_1 = QLabel("Rozmiar: 3")
        row1.addWidget(QLabel("Krok A:")); row1.addWidget(self.combo_morph_1); row1.addWidget(self.slider_morph_1); row1.addWidget(self.lbl_morph_1)
        
        row2 = QHBoxLayout()
        self.combo_morph_2 = QComboBox()
        self.combo_morph_2.addItems(opcje)
        self.slider_morph_2 = QSlider(Qt.Horizontal)
        self.slider_morph_2.setRange(1, 15)
        self.lbl_morph_2 = QLabel("Rozmiar: 3")
        row2.addWidget(QLabel("Krok B:")); row2.addWidget(self.combo_morph_2); row2.addWidget(self.slider_morph_2); row2.addWidget(self.lbl_morph_2)

        self.morph_layout.addLayout(row1)
        self.morph_layout.addLayout(row2)
        layout.addWidget(self.morph_widget)
        self.morph_widget.setVisible(False)
        
        self.combo_morph_1.currentIndexChanged.connect(self.on_morph_changed)
        self.slider_morph_1.valueChanged.connect(self.on_morph_changed)
        self.combo_morph_2.currentIndexChanged.connect(self.on_morph_changed)
        self.slider_morph_2.valueChanged.connect(self.on_morph_changed)

        # --- PANEL MORFOLOGII TĘCZÓWKI
        self.morph_widget_iris = QWidget()
        self.morph_layout_iris = QVBoxLayout(self.morph_widget_iris)
        
        row3 = QHBoxLayout()
        self.combo_morph_3 = QComboBox()
        self.combo_morph_3.addItems(opcje)
        self.slider_morph_3 = QSlider(Qt.Horizontal)
        self.slider_morph_3.setRange(1, 40)
        self.lbl_morph_3 = QLabel("Rozmiar: 3")
        row3.addWidget(QLabel("Krok A (Tęczówka):")); row3.addWidget(self.combo_morph_3); row3.addWidget(self.slider_morph_3); row3.addWidget(self.lbl_morph_3)
        
        row4 = QHBoxLayout()
        self.combo_morph_4 = QComboBox()
        self.combo_morph_4.addItems(opcje)
        self.slider_morph_4 = QSlider(Qt.Horizontal)
        self.slider_morph_4.setRange(1, 40)
        self.lbl_morph_4 = QLabel("Rozmiar: 3")
        row4.addWidget(QLabel("Krok B (Tęczówka):")); row4.addWidget(self.combo_morph_4); row4.addWidget(self.slider_morph_4); row4.addWidget(self.lbl_morph_4)

        self.morph_layout_iris.addLayout(row3)
        self.morph_layout_iris.addLayout(row4)
        layout.addWidget(self.morph_widget_iris)
        self.morph_widget_iris.setVisible(False)
        
        self.combo_morph_3.currentIndexChanged.connect(self.on_morph_iris_changed)
        self.slider_morph_3.valueChanged.connect(self.on_morph_iris_changed)
        self.combo_morph_4.currentIndexChanged.connect(self.on_morph_iris_changed)
        self.slider_morph_4.valueChanged.connect(self.on_morph_iris_changed)
        
        # --- Nawigacja ---
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("← Wstecz")
        self.btn_next = QPushButton("Dalej →")
        self.btn_next.setEnabled(False)
        self.lbl_step = QLabel(f"Krok: {self.current_step} - Wczytywanie")
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_step)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        self.btn_next.clicked.connect(self.next_step)
        self.btn_prev.clicked.connect(self.prev_step)
        
        self.setCentralWidget(main_widget)
        self.resize(800, 600)

    # --- Metody Logiczne UI ---
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz oka", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            pil_img = Image.open(file_path).convert('RGB')
            self.original_image = np.array(pil_img)
            self.current_step = 0
            self.btn_next.setEnabled(True)
            self.process()

    def next_step(self):
        self.current_step += 1
        self.process()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.process()

    def set_controls_visible(self, visible_pupil, visible_iris=False):
        self.lbl_param.setVisible(visible_pupil)
        self.slider_x.setVisible(visible_pupil)
        self.lbl_slider_val.setVisible(visible_pupil)
        self.lbl_param_iris.setVisible(visible_iris)
        self.slider_x_iris.setVisible(visible_iris)
        self.lbl_slider_val_iris.setVisible(visible_iris)

    def set_morph_visible(self, visible):
        self.morph_widget.setVisible(visible)

    def on_slider_changed(self, value):
        self.lbl_slider_val.setText(f"{value / 10.0:.1f}")
        if self.current_step >= 2: self.process()

    def on_slider_iris_changed(self, value):
        self.lbl_slider_val_iris.setText(f"{value / 10.0:.1f}")
        if self.current_step >= 5: self.process()

    def on_morph_changed(self):
        self.lbl_morph_1.setText(f"Rozmiar: {self.slider_morph_1.value() * 2 + 1}")
        self.lbl_morph_2.setText(f"Rozmiar: {self.slider_morph_2.value() * 2 + 1}")
        if self.current_step >= 3: self.process()

    def on_morph_iris_changed(self):
        sz3 = self.slider_morph_3.value() * 2 + 1
        sz4 = self.slider_morph_4.value() * 2 + 1
        self.lbl_morph_3.setText(f"Rozmiar: {sz3}")
        self.lbl_morph_4.setText(f"Rozmiar: {sz4}")
        if self.current_step >= 6: self.process()

    def process(self):
        if self.original_image is None: return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.is_cancelled = True
            self.worker.wait()

        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)

        params = {
            'x_param': self.slider_x.value() / 10.0,
            'x_param_iris': self.slider_x_iris.value() / 10.0,
            'op1': self.combo_morph_1.currentText(),
            'sz1': self.slider_morph_1.value() * 2 + 1,
            'op2': self.combo_morph_2.currentText(),
            'sz2': self.slider_morph_2.value() * 2 + 1,
            # tęczówka
            'op3': self.combo_morph_3.currentText(),
            'sz3': self.slider_morph_3.value() * 2 + 1,
            'op4': self.combo_morph_4.currentText(),
            'sz4': self.slider_morph_4.value() * 2 + 1
        }

        self.worker = IrisWorker(self.original_image, self.current_step, params)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def on_process_finished(self, processed_img):
        self.current_processed_image = processed_img
        self.viewer.wyswietl_obraz_numpy(processed_img)
        
        tytuly_krokow = [
            "Krok 0: Oryginał", "Krok 1: Skala szarości", "Krok 2: Detekcja źrenicy",
            "Krok 3: Morfologia źrenicy", 
            "Krok 4: Środek i promień źrenicy", 
            "Krok 5: Detekcja tęczówki",
            "Krok 6: Morfologia tęczówki",
            "Krok 7: Wyznaczenie promienia tęczówki", 
            "Krok 8: Rozwinięcie tęczówki"
        ]
        self.lbl_step.setText(tytuly_krokow[self.current_step] if self.current_step < len(tytuly_krokow) else f"Krok {self.current_step}")
        
        self.set_controls_visible(visible_pupil=(self.current_step in [2, 3, 4]), visible_iris=(self.current_step in [5, 6, 7]))
        self.morph_widget.setVisible(self.current_step == 3)
        self.morph_widget_iris.setVisible(self.current_step == 6)
        
        if self.current_step == 4:
            self.proj_gora.setVisible(True); self.proj_boczna.setVisible(True)
            gray_img = IrisProcessor.to_grayscale(processed_img)
            inverted = np.where(gray_img < 128, 255, 0).astype(np.uint8)
            self.proj_gora.update_plot(inverted, rgb_mode=False)
            self.proj_boczna.update_plot(inverted, rgb_mode=False)
        else:
            self.proj_gora.setVisible(False); self.proj_boczna.setVisible(False)

        self.btn_prev.setEnabled(self.current_step > 0)
        self.btn_next.setEnabled(True)

    def update_projections_from_rect(self, x, y, w, h):
        if not self.current_processed_image is None and self.current_step == 4:
            widoczny_fragment = self.current_processed_image[y:y+h, x:x+w]
            gray = IrisProcessor.to_grayscale(widoczny_fragment)
            inverted = np.where(gray < 128, 255, 0).astype(np.uint8)
            self.proj_gora.update_plot(inverted, rgb_mode=False)
            self.proj_boczna.update_plot(inverted, rgb_mode=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IrisMainWindow()
    win.show()
    sys.exit(app.exec())
