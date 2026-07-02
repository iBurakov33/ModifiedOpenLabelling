"""
Module for range labeling of objects on image sequences.
Allows saving a labeling template and applying it to a range of frames.

Hotkeys:
    [Space] - Save current rectangle as template
    [f]     - Set first frame of range
    [g]     - Set last frame of range
    [b]     - Apply template to range
    [c]     - Reset all settings
"""

import cv2
import numpy as np
import os


class RangeLabeler:
    def __init__(self, with_qt=True):
        self.with_qt = with_qt
        self.template_bbox = None
        self.template_class = 0
        self.template_track = 0
        self.selection_start = -1
        self.selection_end = -1
        self.is_active = False
        self.status_color = (0, 255, 255)
        self.bg_color = (0, 0, 0)

        
    def save_template(self, bbox_data, class_idx, track_idx=0):
        self.template_bbox = bbox_data
        self.template_class = class_idx
        self.template_track = track_idx
        self.is_active = True
        self._show_message(f"[TEMPLATE] Saved! Class: {class_idx}, Track: {track_idx}")
        
    def set_first_frame(self, frame_idx):
        if not self.is_active or self.template_bbox is None:
            self._show_message("[WARN] First save template (Space)!")
            return False
        self.selection_start = frame_idx
        self._show_message(f"[FIRST] First frame: {frame_idx}")
        return True
        
    def set_last_frame(self, frame_idx):
        if not self.is_active or self.template_bbox is None:
            self._show_message("[WARN] First save template (Space)!")
            return False
        self.selection_end = frame_idx
        self._show_message(f"[LAST] Last frame: {frame_idx}")
        return True
        
    def apply_to_range(self, start_idx, end_idx, image_list, get_txt_path_func, save_bb_func):
        if not self.is_active or self.template_bbox is None:
            self._show_message("[WARN] No template saved!")
            return False
            
        if start_idx == -1 or end_idx == -1:
            self._show_message("[WARN] Set first (f) and last (g) frame!")
            return False
            
        start = min(start_idx, end_idx)
        end = max(start_idx, end_idx)
        
        if start < 0 or end >= len(image_list):
            self._show_message(f"[WARN] Index out of range (0-{len(image_list)-1})!")
            return False
            
        x1, y1, x2, y2 = self.template_bbox
        class_idx = self.template_class
        track_idx = self.template_track
        
        success_count = 0
        total_frames = end - start + 1
        
        for i in range(start, end + 1):
            img_path = image_list[i]
            print(f"[DEBUG] img_path{img_path} ")
            img = cv2.imread(img_path)
            if img is None:
                self._show_message(f"[WARN] Failed to load frame {i}: {img_path}")
                continue
                
            height, width = img.shape[:2]
            txt_path = get_txt_path_func(img_path)
            print(f"[DEBUG] txt_path{txt_path} ")

            x_center = (x1 + x2) / (2.0 * width)
            y_center = (y1 + y2) / (2.0 * height)
            bbox_width = abs(x2 - x1) / width
            bbox_height = abs(y2 - y1) / height
            
            line = f"{class_idx} {track_idx} {x_center} {y_center} {bbox_width} {bbox_height}"
            save_bb_func(txt_path, line)
            success_count += 1
            
            if i % 10 == 0 or i == end:
                progress = ((i - start + 1) / total_frames) * 100
                self._show_message(f"[PROGRESS] {i - start + 1}/{total_frames} ({progress:.0f}%)")
        
        self._show_message(f"[OK] Applied to {success_count} frames (from {start} to {end})")
        return True
        
    def reset(self):
        self.template_bbox = None
        self.template_class = 0
        self.template_track = 0
        self.selection_start = -1
        self.selection_end = -1
        self.is_active = False
        self._show_message("[RESET] State reset")
        
    def get_status_text(self):
        if not self.is_active or self.template_bbox is None:
            return ""
            
        status = f"TEMPLATE: class {self.template_class}"
        
        if self.selection_start != -1:
            status += f" | FIRST: {self.selection_start}"
        if self.selection_end != -1:
            status += f" | LAST: {self.selection_end}"
            
        if self.selection_start != -1 and self.selection_end != -1:
            count = abs(self.selection_end - self.selection_start) + 1
            status += f" | RANGE: {count} frames"
            
        return status
        
    def draw_status(self, image):
        status_text = self.get_status_text()
        if not status_text:
            return image
            
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (500, 35), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        cv2.putText(image, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.status_color, 2)
        return image
        
    def _show_message(self, message):
        if self.with_qt:
            cv2.displayOverlay("Bounding Box Labeler", message, 2000)
        else:
            print(f"[RangeLabeler] {message}")

    def delete_template_from_range(self, start_idx, end_idx, image_list, get_txt_path_func, delete_bb_func, threshold=0.0):
        """
        Удаляет из диапазона все прямоугольники, совпадающие с шаблоном.
        
        Args:
            start_idx: int - начальный кадр
            end_idx: int - конечный кадр
            image_list: list - список изображений
            get_txt_path_func: function - получить путь к .txt
            delete_bb_func: function - удалить bounding box
            threshold: float - порог совпадения (0.1 = 10%)
        """
        if not self.is_active or self.template_bbox is None:
            self._show_message("[WARN] No template saved! Save template with Space first!")
            return False
            
        if start_idx == -1 or end_idx == -1:
            self._show_message("[WARN] Set first (f) and last (g) frame first!")
            return False
            
        start = min(start_idx, end_idx)
        end = max(start_idx, end_idx)
        

        template_class = self.template_class
        tx1, ty1, tx2, ty2 = self.template_bbox
        tw = abs(tx2 - tx1)
        th = abs(ty2 - ty1)
        
        deleted_count = 0
        total_frames = end - start + 1
        
        for i in range(start, end + 1):
            img_path = image_list[i]
            txt_path = get_txt_path_func(img_path)
            
            if not os.path.exists(txt_path):
                continue
                
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                values = line.strip().split()
                if len(values) < 5:
                    new_lines.append(line)
                    continue
                    
                # Парсим строку (YOLO формат: class track_id x_center y_center width height)
                class_idx = int(float(values[0]))
                
                # Если класс не совпадает - сохраняем
                if class_idx != template_class:
                    new_lines.append(line)
                    continue
                    
                # Проверяем координаты (для YOLO формата)
                x_center = float(values[2])
                y_center = float(values[3])
                bbox_width = float(values[4])
                bbox_height = float(values[5])
                
                # Получаем размеры изображения
                img = cv2.imread(img_path)
                if img is None:
                    new_lines.append(line)
                    continue
                height, width = img.shape[:2]
                
                x1 = int((x_center - bbox_width/2) * width)
                y1 = int((y_center - bbox_height/2) * height)
                x2 = int((x_center + bbox_width/2) * width)
                y2 = int((y_center + bbox_height/2) * height)
                
                # Проверяем совпадение с шаблоном (по положению)
                # Вычисляем IoU (Intersection over Union) или просто разницу
                # Здесь просто проверяем, что координаты близки
                diff_x = abs(x1 - tx1) / max(width, 1)
                diff_y = abs(y1 - ty1) / max(height, 1)
                diff_w = abs((x2 - x1) - tw) / max(tw, 1)
                diff_h = abs((y2 - y1) - th) / max(th, 1)
                
                print (f"[debug]{diff_x}\t{diff_h}\t {diff_w}\t {diff_y}")
                # Если все отклонения меньше порога - удаляем
                if threshold == 1.0 or (diff_x <= threshold and diff_y <= threshold and diff_w <= threshold and diff_h <= threshold):
                    deleted_count += 1
                    print(f"[DEBUG] Deleted bbox in frame {i}: class {class_idx}")
                    continue  
                else:
                    new_lines.append(line)
            
            with open(txt_path, 'w') as f:
                f.writelines(new_lines)
            
            # Показываем прогресс
            if i % 10 == 0 or i == end:
                progress = ((i - start + 1) / total_frames) * 100
                self._show_message(f"[PROGRESS] {i - start + 1}/{total_frames} ({progress:.0f}%)")
        
        self._show_message(f"[OK] Deleted {deleted_count} matching bboxes from {total_frames} frames")
        return True
    
def create_range_labeler_help():
    return """
    === RANGE LABELING ===
    [Space] Save current rectangle as template
    [f]     Set first frame of range
    [g]     Set last frame of range
    [b]     Apply template to range
    [z]     Delete template to range
    [+]     Increase threshold"
    [-]     Decrease threshold"
    [c]     Reset all settings
    """