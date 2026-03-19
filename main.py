import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Gọi mô hình 
model = YOLO('best.pt')

# 2. Hàm xử lý ẢNH TĨNH
def xu_ly_anh(anh_dau_vao):
    if anh_dau_vao is None:
        return None
        
    # Chuyển màu RGB sang BGR 
    anh_bgr = cv2.cvtColor(anh_dau_vao, cv2.COLOR_RGB2BGR)
    
    # Nhận diện conf 0.20
    results = model.predict(anh_bgr, conf=0.20, verbose=False)
    
    # In ra Terminal
    so_luong = len(results[0].boxes)
    print(f"-> Đã nhận diện được: {so_luong} vật thể trong bức ảnh này.")
    
    # Vẽ Bounding Box
    anh_da_ve = results[0].plot()
    
    # Chuyển ngược lại sang RGB 
    anh_rgb_xuat_ra = cv2.cvtColor(anh_da_ve, cv2.COLOR_BGR2RGB)
    
    return anh_rgb_xuat_ra

# 3. Hàm xử lý VIDEO 
# def xu_ly_video(duong_dan_video):
#     duong_dan_xuat = "ket_qua_nhan_dien.mp4"
#     cap = cv2.VideoCapture(duong_dan_video)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(duong_dan_xuat, fourcc, fps, (width, height))
    
#     print("-> Đang xử lý video... Vui lòng đợi.")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
            
#         results = model.predict(frame, conf=0.20, verbose=False)
#         out.write(results[0].plot())

#     cap.release()
#     out.release()
#     print("-> Xử lý video hoàn tất!")
    
#     return duong_dan_xuat

# Hàm xử lý video + đếm 
def xu_ly_video(duong_dan_video):
    if duong_dan_video is None:
        return None
        
    duong_dan_xuat = "ket_qua_dem_san_luong.mp4"
    cap = cv2.VideoCapture(duong_dan_video)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Xuất video MP4
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(duong_dan_xuat, fourcc, fps, (width, height))
    
    # LOGIC ĐẾM 
    # Tọa độ vạch kẻ
    # toa_do_vach_y = int(height * 0.8)
    
    # lich_su_vi_tri = {}   # Lưu tọa độ cũ của từng quả táo: {id_qua_tao: toa_do_y_cu}
    # danh_sach_da_dem = [] # Lưu danh sách các quả đã rơi qua vạch (chống đếm trùng)
    
    # print("-> [Video] Đang khởi động hệ thống đếm sản lượng...")
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret: break
        
    #     # Gọi YOLO KÈM THEO BYTETRACK (model.track thay vì model.predict)
    #     results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
    #     anh_da_ve = results[0].plot()
        
    #     # Vẽ vạch ranh giới đếm (Màu đỏ, nét dày 3)
    #     cv2.line(anh_da_ve, (0, toa_do_vach_y), (width, toa_do_vach_y), (0, 0, 255), 3)
        
    #     # Bắt đầu xử lý logic đếm nếu có vật thể và có ID được gán
    #     if results[0].boxes is not None and results[0].boxes.id is not None:
    #         boxes = results[0].boxes.xyxy.cpu().numpy()
    #         track_ids = results[0].boxes.id.int().cpu().numpy()
            
    #         for box, track_id in zip(boxes, track_ids):
    #             x1, y1, x2, y2 = box
                
    #             # Tính tọa độ Tâm (Centroid) của quả táo
    #             tam_x = int((x1 + x2) / 2)
    #             tam_y = int((y1 + y2) / 2)
                
    #             # Vẽ chấm xanh lá cây ở tâm vật thể
    #             cv2.circle(anh_da_ve, (tam_x, tam_y), 5, (0, 255, 0), -1)
                
    #             # LOGIC VƯỢT VẠCH (Line Crossing)
    #             if track_id in lich_su_vi_tri:
    #                 tam_y_cu = lich_su_vi_tri[track_id]
                    
    #                 # Nếu frame trước nó ở TRÊN vạch, frame này nó rớt XUỐNG DƯỚI vạch
    #                 if tam_y_cu < toa_do_vach_y and tam_y >= toa_do_vach_y:
    #                     if track_id not in danh_sach_da_dem:
    #                         danh_sach_da_dem.append(track_id) # Chốt sổ, cộng 1 vào tổng
                            
    #             # Cập nhật lại vị trí hiện tại để frame sau so sánh tiếp
    #             lich_su_vi_tri[track_id] = tam_y
                
    #     # Hiển thị số lượng Tổng sản lượng lên màn hình video
    #     tong_san_luong = len(danh_sach_da_dem)
    #     cv2.putText(anh_da_ve, f"Tong san luong: {tong_san_luong}", (30, 70), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        
    #     out.write(anh_da_ve)
    
    # Dùng set() của Python để lưu ID. Set tự động loại bỏ các ID trùng lặp!
    danh_sach_id_da_dem = set() 
    
    print("-> [Video] Đang khởi động đếm tổng vật thể bằng ByteTrack...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Gọi YOLO và ByteTrack
        #results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, iou=0.65, verbose=False)
        anh_da_ve = results[0].plot()
        
        # Nếu phát hiện vật thể và ByteTrack đã cấp ID
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Lấy toàn bộ ID có trong khung hình hiện tại
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Quét qua từng ID và ném vào rổ set()
            for track_id in track_ids:
                danh_sach_id_da_dem.add(track_id)
                
        # Tổng sản lượng chính là số lượng ID độc nhất có trong rổ
        tong_san_luong = len(danh_sach_id_da_dem)
        
        # In kết quả lên góc trái màn hình
        cv2.putText(anh_da_ve, f"Tong san luong: {tong_san_luong}", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        out.write(anh_da_ve)

    cap.release()
    out.release()
    print(f"-> [Video] Xử lý hoàn tất! Đã đếm được: {len(danh_sach_id_da_dem)} vật thể.")
    return duong_dan_xuat
# 4. Web 
with gr.Blocks(theme=gr.themes.Soft()) as giao_dien:
    gr.Markdown("<h1 style='text-align: center;'>Hệ Thống Tự Động Nhận Diện Trái Cây</h1>")
    gr.Markdown("**Sinh viên thực hiện:** Khiếu Hữu Tiến Dũng")
    
    with gr.Tabs():
        # TAB 1: Ảnh
        with gr.TabItem("Nhận diện Ảnh tĩnh"):
            with gr.Row():
                dau_vao_anh = gr.Image(label="Tải ảnh lên đây", type="numpy")
                dau_ra_anh = gr.Image(label="Kết quả nhận diện")
            nut_chay_anh = gr.Button("Bắt đầu nhận diện Ảnh", variant="primary")
            nut_chay_anh.click(fn=xu_ly_anh, inputs=dau_vao_anh, outputs=dau_ra_anh)
            
        # TAB 2: Video
        with gr.TabItem("Nhận diện Video"):
            with gr.Row():
                dau_vao_vid = gr.Video(label="Tải video lên đây")
                dau_ra_vid = gr.Video(label="Kết quả")
            nut_chay_vid = gr.Button("Bắt đầu nhận diện Video", variant="primary")
            nut_chay_vid.click(fn=xu_ly_video, inputs=dau_vao_vid, outputs=dau_ra_vid)

# 5. Chạy
if __name__ == "__main__":
    giao_dien.launch()