import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO
#from torchreid.reid.utils.feature_extractor import FeatureExtractor
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import collections
import time
# from deep_sort_realtime.deepsort_tracker import DeepSort

# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH
#     boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
#     boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_center_distance(boxA, boxB):
    """Compute Euclidean distance between box centers."""
    # box = (x1, y1, x2, y2)
    cxA, cyA = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
    cxB, cyB = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
    return np.hypot(cxA - cxB, cyA - cyB)

def compute_box_distance(boxA, boxB):
    """
    計算兩個軸平行邊界框之間的最短距離（像素為單位）。
    如果重疊則距離為 0。
    
    參數：
        boxA, boxB: tuple (x1, y1, x2, y2)，左上角與右下角座標
    回傳：
        float，最短歐氏距離
    """
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # 計算水平與垂直方向的距離（若重疊則為 0）
    dx = max(xB1 - xA2, xA1 - xB2, 0)
    dy = max(yB1 - yA2, yA1 - yB2, 0)

    return np.hypot(dx, dy)  # √(dx² + dy²)

class PersonReIDNode(Node):
    def __init__(self):
        super().__init__('person_reid_node')
        self.bridge = CvBridge()

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("similarity_threshold", 0.6)

        image_topic = self.get_parameter("image_topic").value
        self.similarity_threshold = self.get_parameter("similarity_threshold").value

        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        self.create_subscription(PointStamped, '/clicked_point', self.point_callback, 10)

        self.image_pub = self.create_publisher(Image, '/person_reid/image', 10)
        self.bbox_pub = self.create_publisher(Int32MultiArray, '/person_reid/bboxes', 10)
        
        
        # instantiate DeepSort once
        # self.tracker = DeepSort(max_age=5,  embedder='torchreid', nms_max_overlap=0.7)
    
        # 用來暫存每幀的 DeepSort tracks
        self.current_tracks = []          # List[tuple(track_id, (x1,y1,x2,y2))]
        # 用來記你滑鼠點了哪個 DeepSort ID
        self.tracked_deep_id = -1         # List[int] list of selected track IDs
        
        # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512', 'se_resnet50', 'se_resnet50_fc512', 'se_resnet101',
        #  'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512', 
        #  'inceptionresnetv2', 'inceptionv4', 'xception', 'resnet50_ibn_a', 'resnet50_ibn_b', 'nasnsetmobile', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 
        #  'shufflenet', 'squeezenet1_0', 'squeezenet1_0_fc512', 'squeezenet1_1', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        #  'mudeep', 'resnet50mid', 'hacnn', 'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 
        #  'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25']"


        # Models
        # self.detector = YOLO("yolov8n.pt", verbose=True)
        # self.detector = YOLO("yolov8s.pt", verbose=True)
        self.detector = YOLO("yolo11n.pt", verbose=True)
        # self.trt_model = YOLO("yolo11n.engine")
        # self.detector = YOLO(" yolov8n-seg.pt", verbose=True)
        self.extractor = FeatureExtractor(
            # model_name="osnet_x1_0",
            model_name="osnet_ain_x1_0",
            # model_path="osnet_ain_x1_0_imagenet.pth",
            model_path="/root/ros2_ws/src/reid_tracker/models/osnet_ain_x1_0_ms_d_c.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True
        )
        # self.extractor = FeatureExtractor(
        #     model_name="osnet_ain_x1_0",
        #     model_path="osnet_ain_x1_0_imagenet.pth",
        #     device="cuda" if torch.cuda.is_available() else "cpu",
        #     verbose=False
        # )
    

        # ReID memory
        self.gallery = []        # List[np.ndarray]
        self.gallery_ids = []    # List[int]
        self.next_id = 1         # start ID
        self.tracked_ids = []    # 可同時追蹤多個
        self.latest_boxes = []   # [(x1, y1, x2, y2), feature]
        self.sims = []
        # self.person_ids = []
        
        self.tracked_person_feature = None
        
        self.maxlen_dq = 60
        
        self.last_trk_box = None     # updated on click or on match
        #self.skip_iou_thresh = 0.2  # you can tune this
        self.skip_dist_thresh = 10   # distance threshold to skip feature extraction
              
        ## Timer
        self.last_time = time.time()
        
        
        # Setup GUI and mouse click
        cv2.namedWindow("Person ReID", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Person ReID", self.mouse_callback)
        

        self.get_logger().info("Person ReID node initialized.") 
        
    
    def image_callback(self, msg):
        print(f'!!!!!!!!self.gallery_ids: {self.gallery_ids}')
        print(f'!!!!!!!!self.tracked_ids: {self.tracked_ids}')
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        vis_frame = frame.copy()

        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        # === 顯示 FPS ===
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        results = self.detector(frame)[0] 
        # results = self.trt_model(frame)[0] 
        # person_boxes = [r for r in results.boxes if int(r.cls) == 0] # 只擷取行人
        confidence_threshold = 0.3 #0.65
        person_boxes = [
            r for r in results.boxes
            if int(r.cls) == 0 and float(r.conf[0]) >= confidence_threshold
        ]

        # --- 1. Run DeepSort tracking ---
        # dets = []
        # for r in person_boxes:
        #     x1, y1, x2, y2 = map(int, r.xyxy[0])
        #     conf = float(r.conf[0])
        #     dets.append(((x1, y1, x2-x1, y2-y1), conf, 0)) # DeepSort 要的是 (x, y, w, h)
            
        # self.current_tracks = self.tracker.update_tracks(dets, frame=frame)
        # for tr in self.current_tracks:
        #     if not tr.is_confirmed(): continue
        #     tid = tr.track_id # 只畫你點擊過的那個 ID
        #     l, t, r, b = tr.to_ltrb()
        #     bbox = (int(l), int(t), int(r), int(b))
        #     if tid == self.tracked_deep_id:
        #         # 被點選追蹤的紅框
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        #         cv2.putText(frame, f"Deep ID:{tid}", (bbox[0], bbox[1] - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #     else:
        #         # 其他 track 用灰色框
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200, 200, 200), 2)
        #         cv2.putText(frame, f"Deep ID:{tid}", (bbox[0], bbox[1] - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
        # --- 2. Your existing ReID feature logic can go here --

        boxes_msg = Int32MultiArray()
        
        # 先擷取所有行人特徵
        self.latest_boxes.clear()
        for r in person_boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            # Resize to 256x128 (H x W)
            crop_resized = cv2.resize(crop, (128, 256))  # OpenCV: (width, height)
            # 提取特徵並 normalize
            feat = self.extractor(crop_resized)  # 由偵測框提取特徵                 # Torch tensor, shape e.g. (1, C)
            print(feat)
            feat_np = feat.cpu().detach().numpy().squeeze()# → shape (C,)
            feat_np = feat_np / np.linalg.norm(feat_np)    # L2 normalize
            self.latest_boxes.append(((x1, y1, x2, y2), feat_np, crop))
            
  
        match_candidates = []
        for i, ((x1, y1, x2, y2), feat_np, crop) in enumerate(self.latest_boxes):
            if self.gallery: # 若 gallery 有東西就算相似度
                # print(f'The len of gallery[0]: {len(self.gallery[0])}') ## gallery 永遠只會有一個東西
                # 先從 deque 中取出所有字典裡的 feat 向量，組成一個 list
                gallery_feats = [d['feat'] for d in self.gallery[0]]  # self.gallery[0] 是個 deque，裡面存的是 dict
                if not gallery_feats: break  # 沒東西就跳出
                gallery_matrix = np.stack(gallery_feats, axis=0) # 再把這些 (C,) 向量堆疊成 (N, C) 的矩陣
                sims = cosine_similarity(feat_np.reshape(1, -1), gallery_matrix)[0]
                # sims = cosine_similarity(feat_np.reshape(1, -1), np.stack(self.gallery[0][0]['feat'], axis=0))[0]
                best_score = float(sims.max())  # 直接取 deque 內歷史特徵的最高分
                if best_score >= self.similarity_threshold:
                    match_candidates.append({
                        'index': i,
                        'score': best_score,
                        'box': (x1, y1, x2, y2),
                        'feat': feat_np,
                        'crop': crop
                    })
                else:
                    self.last_trk_box = None
        
        # if self.last_trk_box  and (compute_box_distance((x1, y1, x2, y2), self.last_trk_box)<100) :
        #     pass
                    
        # 2. 從候選者中找出最相似且分數最高的那個
        best_match_idx = -1
        #best_match_score = -1
        #best_match_box = None
        # 暫存所有人框與其相似度
        if match_candidates:
            best_candidate = max(match_candidates, key=lambda c: c['score'])  # 直接用 max 選擇 score 最高
            best_match_idx  = best_candidate['index']
            #best_match_box  = best_candidate['box']
            #best_match_feat = best_candidate['feat']
            #best_match_score = best_candidate['score']
            

        # 然後重新遍歷 boxes 並只畫最相似的那個人為紅框
        for i, ((x1, y1, x2, y2), feat_np, crop) in enumerate(self.latest_boxes):
            # 1) 預設 unmatched
            person_id = -1 
            color = (200, 200, 200) # 灰色框表示未比對

            if i == best_match_idx:
                person_id = self.gallery_ids[0]
                color = (0, 0, 255)
                # self.gallery[0].append(best_candidate['feat'])
                self.gallery[0].append({'feat': best_candidate['feat'], 'crop': best_candidate['crop']})
                self.last_trk_box = (x1, y1, x2, y2)
                print(f"[BEST SELECT] Box {(x1,y1,x2,y2)} score={best_candidate['score']}")
            else:
                # 若非最佳 match，可以畫灰色或綠色（例如太靠近）
                if self.last_trk_box is not None:
                    dist = compute_box_distance((x1, y1, x2, y2), self.last_trk_box)
                    if dist < self.skip_dist_thresh:
                        color = (0, 255, 0)

            boxes_msg.data.extend([x1, y1, x2, y2, person_id])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"ID:{person_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5. 顯示整個 gallery 的裁切影像 # 將 deque 內的東西以圖像的方式秀出來
        max_display = 5
        y_offset = 10        # 所有視窗的垂直位置
        
        if self.gallery:
            # 1. 取最多前 5 張條形圖
            # max_display = min(5, len(self.gallery[0]))
            # 2. 每張 resize 後的條形圖高度=10, 寬度=512
            bar_height, bar_width = 512, 40
           # 3. 建空畫布：高度10, 寬度＝bar_width * max_display, 通道數同原圖
            # 假設 entry 是 BGR ndarray
            # sample = self.gallery[0][0]
            # canvas = np.zeros((bar_height,
            #                 bar_width * 5 + y_offset*4),
            #                 dtype=sample.dtype)
            canvas = np.zeros((bar_height, bar_width * 5 + y_offset*4, 3), dtype=np.uint8)


            # 4. 將每個條形圖貼到 canvas 上
            # for i, dict in enumerate(self.gallery[0]):
            #     if i >= max_display: break
            #     else:
            #         # 如果 gallery 裡存的是 (feat, crop) tuple 就先 unpack:
            #         # _, crop = entry
            #         # feat = entry  # or unpacked crop
            #         feat = dict['feat']
            #         crop = dict['crop']
            #         norm = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #         color_bar = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            #         # bar = cv2.resize(feat, (bar_width, bar_height))
            #         bar = cv2.resize(color_bar, (bar_width, bar_height))
            #         x_start = i * (bar_width+y_offset) 
            #         canvas[:, x_start:x_start+bar_width] = bar
            
            # 假設在前面已有：
            # max_display, bar_width, bar_height, y_offset, self.gallery, self.maxlen_dq
            # 且 gallery[0] 裡面每個 entry 是 {'feat':…, 'crop':…}

            # 1. 計算 canvas 大小
            crop_height = 150  # 你希望顯示的 crop 高度
            total_width = max_display * (bar_width + y_offset)
            canvas_height = crop_height + bar_height
            canvas = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)

            # 2. 把每個 crop 與對應條形圖貼上去
            for i, entry in enumerate(reversed(self.gallery[0])):
                if i >= max_display: break

                feat = entry['feat']
                crop = entry['crop']

                # resize crop 到 bar_width x crop_height
                # resized_crop = cv2.resize(crop, (bar_width, crop_height))
                                # resize crop 到 bar_width x crop_height (保留長寬比，並置中)
                h, w = crop.shape[:2]
                scale = min(bar_width / w, crop_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(crop, (new_w, new_h))
                # 建立背景，大小為 bar_width x crop_height
                resized_crop = np.zeros((crop_height, bar_width, 3), dtype=np.uint8)
                # 計算置中偏移
                x_off = (bar_width - new_w) // 2
                y_off = (crop_height - new_h) // 2
                # 將保持比例後的 crop 貼到背景中央
                resized_crop[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                # normalize + colormap 再 resize 條形圖
                norm = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # color_bar = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                color_bar = cv2.applyColorMap(norm, cv2.COLORMAP_SUMMER)
                bar = cv2.resize(color_bar, (bar_width, bar_height))

                x_start = i * (bar_width + y_offset)
                x_end   = x_start + bar_width

                # 3. 上半部放 crop
                canvas[0:crop_height, x_start:x_end] = resized_crop
                # 4. 下半部放 feature 條形圖
                canvas[crop_height:crop_height+bar_height, x_start:x_end] = bar

            # 5. 顯示在同一個視窗
            cv2.imshow('ReID Gallery', canvas)

        self.bbox_pub.publish(boxes_msg)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis_frame, 'bgr8'))
        cv2.imshow('Person ReID', vis_frame)
        cv2.waitKey(1)

        
        
        # for (x1, y1, x2, y2), feat_np in self.latest_boxes:
        #     # 1) 預設 unmatched
        #     person_id = -1
        #     color     = (200,200,200)  # 灰色框表示未比對

        #     # 2) 若 gallery 有東西就算相似度
        #     if self.gallery:
        #         # print(f'gallary:{self.gallery[0][0]}')
        #         sims = cosine_similarity(feat_np.reshape(1, -1),np.stack(self.gallery[0], axis=0))[0]  # e.g. array([0.82, 0.45, …])
                
        #         best_idx   = int(np.argmax(sims))
        #         # best_score = float(sims[best_idx])
        #         best_score = float(sims.max())          # 直接取 deque 內歷史特徵的最高分
                
        #         # 3) 門檻判斷
        #         if best_score >= self.similarity_threshold:
        #             # person_id = self.gallery_ids[best_idx]
        #             person_id = self.gallery_ids[0]
        #             color = (0,0,255)  # 紅色框表示有 match
        #             # if len(self.gallery[0]) < self.maxlen_dq: # 只在少於 5 筆時才加入，保留最初五幀
        #             self.gallery[0].append(feat_np) #把當前特徵加進 deque，deque 會自動丟掉最舊的一筆
        #             # update last_trk_box for next-frame IoU skip
        #             self.last_trk_box = (x1, y1, x2, y2)
        #         else:
        #              # —— NEW: skip any detection too close to last tracked box —— 
        #             curr_box = (x1, y1, x2, y2)
        #             if self.last_trk_box is not None:
        #                 dist = compute_box_distance(curr_box, self.last_trk_box)
        #                 if  dist < 10: # too close to tracked person, don't extract or add
        #                     color   = (0,255,0)  # GREEN
                    
        #         # debug 印出每框的相似度
        #         print(f"Box {(x1,y1,x2,y2)} sims={sims}, best={best_score}, id={person_id}")


        #     # 4) 把這一框的結果放到 message 裡
        #     boxes_msg.data.extend([x1, y1, x2, y2, person_id])
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(
        #         frame, f"ID:{person_id}", (x1, y1 - 5),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
        #     )

        # self.bbox_pub.publish(boxes_msg)
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        # cv2.imshow('Person ReID', frame)
        # cv2.waitKey(1)
        
    def add_to_gallery(self, feat_np_dq):
        """把新的特徵加入 gallery，回傳這個 feat 的 ID。"""
        self.gallery.append(feat_np_dq)
        self.gallery_ids.append(self.next_id)
        self.next_id += 1
        return self.gallery_ids[-1]
    
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
            
        # 1) 優先用 DeepSort track 點擊選人
        # for tr in self.current_tracks:
        #     if not tr.is_confirmed(): continue
        #     # 取得這個 track 的 bbox
        #     l, t, r, b = map(int, tr.to_ltrb())
        #     # 如果點擊位置落在此 bbox 內
        #     if l <= x <= r and t <= y <= b:
        #         tid = tr.track_id
        #         # 更新被追蹤的 DeepSort ID
        #         self.tracked_deep_id = tid
        #         self.get_logger().info(f"Selected DeepSort track ID {tid}")
        #         return # 命中後直接結束，不做後續 gallery 邏輯
            


        for (x1, y1, x2, y2), feat_np, crop in self.latest_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Clear gallery → only keep selected person
                self.gallery.clear()
                self.gallery_ids.clear()
                self.tracked_ids.clear()

                # self.gallery.append(feat_np)
                # self.gallery_ids.append(0)
                # self.tracked_ids.append(0)
                dq = collections.deque(maxlen=self.maxlen_dq)
                dq.append({'feat': feat_np, 'crop': crop})
                new_id = self.add_to_gallery(dq)
                # new_id = self.add_to_gallery(feat_np)
                self.tracked_ids.append(new_id)
                self.last_trk_box = (x1, y1, x2, y2)

                self.get_logger().info(f"Selected person at ({x}, {y}) for tracking.")
                self.get_logger().info(f"Added person ID {new_id} to gallery")
                break
    
    # def mouse_callback(self, event, x, y, flags, param):
    #     if event != cv2.EVENT_LBUTTONDOWN: return
    #     # 點選之後把那格的特徵加入 gallery，並記錄這個 ID
    #     for (x1, y1, x2, y2), feat_np in self.latest_boxes:
    #         if x1 <= x <= x2 and y1 <= y <= y2:
    #             new_id = self.add_to_gallery(feat_np)
    #             self.tracked_ids.append(new_id)
    #             self.get_logger().info(f"Added person ID {new_id} to gallery")
    #             break
    
                
    def point_callback(self, msg):
        x, y = int(msg.point.x), int(msg.point.y)
        for (x1, y1, x2, y2), feat in self.latest_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.get_logger().info(f"Selected person via RViz at ({x},{y})")
                self.tracked_person_feature = feat
                return
    
    # def point_callback(self, msg):
    #     x, y = int(msg.point.x), int(msg.point.y)
    #     for (x1, y1, x2, y2), feat_np in self.latest_boxes:
    #         if x1 <= x <= x2 and y1 <= y <= y2:
    #             self.gallery.clear()
    #             self.gallery_ids.clear()
    #             self.tracked_ids.clear()

    #             self.gallery.append(feat_np)
    #             self.gallery_ids.append(0)
    #             self.tracked_ids.append(0)

    #             self.get_logger().info(f"Selected person from RViz at ({x},{y})")
    #             return


def main(args=None):
    rclpy.init(args=args)
    node = PersonReIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

