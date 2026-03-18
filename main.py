import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Gọi mô hình YOLOv26-s lên
model = YOLO('best.pt')

# 2. Hàm xử lý ẢNH TĨNH
def xu_ly_anh(anh_dau_vao):
    if anh_dau_vao is None:
        return None
        
    # Chuyển màu RGB (Gradio) sang BGR (YOLO/OpenCV) để không bị lỗi màu
    anh_bgr = cv2.cvtColor(anh_dau_vao, cv2.COLOR_RGB2BGR)
    
    # Ép AI nhận diện với mức độ tự tin (20%)
    results = model.predict(anh_bgr, conf=0.20, verbose=False)
    
    # In thông báo ra Terminal
    so_luong = len(results[0].boxes)
    print(f"-> Đã nhận diện được: {so_luong} vật thể trong bức ảnh này.")
    
    # Vẽ khung Bounding Box
    anh_da_ve = results[0].plot()
    
    # Chuyển ngược lại sang RGB để web hiển thị đúng màu
    anh_rgb_xuat_ra = cv2.cvtColor(anh_da_ve, cv2.COLOR_BGR2RGB)
    
    return anh_rgb_xuat_ra

# 3. Hàm xử lý VIDEO CHUYỂN ĐỘNG
def xu_ly_video(duong_dan_video):
    duong_dan_xuat = "ket_qua_nhan_dien.mp4"
    cap = cv2.VideoCapture(duong_dan_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(duong_dan_xuat, fourcc, fps, (width, height))
    
    print("-> Đang xử lý video... Vui lòng đợi.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        results = model.predict(frame, conf=0.20, verbose=False)
        out.write(results[0].plot())

    cap.release()
    out.release()
    print("-> Xử lý video hoàn tất!")
    
    return duong_dan_xuat

# 4. Giao diện Web (UI)
with gr.Blocks(theme=gr.themes.Soft()) as giao_dien:
    gr.Markdown("<h1 style='text-align: center;'>Hệ Thống Tự Động Nhận Diện Trái Cây</h1>")
    gr.Markdown("**Sinh viên thực hiện:** Khiếu Hữu Tiến Dũng")
    
    with gr.Tabs():
        # TAB 1: Dành cho Ảnh
        with gr.TabItem("Nhận diện Ảnh tĩnh"):
            with gr.Row():
                dau_vao_anh = gr.Image(label="Tải ảnh lên đây", type="numpy")
                dau_ra_anh = gr.Image(label="Kết quả nhận diện")
            nut_chay_anh = gr.Button("Bắt đầu nhận diện Ảnh", variant="primary")
            nut_chay_anh.click(fn=xu_ly_anh, inputs=dau_vao_anh, outputs=dau_ra_anh)
            
        # TAB 2: Dành cho Video
        with gr.TabItem("Nhận diện Video"):
            with gr.Row():
                dau_vao_vid = gr.Video(label="Tải video lên đây")
                dau_ra_vid = gr.Video(label="Kết quả nhận diện")
            nut_chay_vid = gr.Button("Bắt đầu nhận diện Video", variant="primary")
            nut_chay_vid.click(fn=xu_ly_video, inputs=dau_vao_vid, outputs=dau_ra_vid)

# 5. Khởi chạy
if __name__ == "__main__":
    giao_dien.launch()