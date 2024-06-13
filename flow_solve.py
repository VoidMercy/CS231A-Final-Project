import cv2
import random
import numpy as np
from tqdm import trange

video_cap = cv2.VideoCapture("v2.mp4")

length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
last_pos = None
count = 0

positions = [
    ('a', 74.5, 202.5),
    ('b', 118.5, 172.5),
    ('c', 161.5, 155.5),
    ('d', 202.5, 139.5),
    ('e', 247.5, 135.5),
    ('f', 289.5, 128.5),
    ('g', 329.5, 124.5),
    ('h', 382.5, 130.5),
    ('i', 420.5, 137.5),
    ('j', 453.5, 140.5),
    ('k', 496.5, 153.5),
    ('l', 545.5, 174.5),
    ('m', 595.5, 200.5),
    ('n', 65.5, 297.5),
    ('o', 116.5, 270.5),
    ('p', 146.5, 241.5),
    ('q', 196.5, 222.5),
    ('r', 243.5, 207.5),
    ('s', 289.5, 201.5),
    ('t', 333.5, 193.5),
    ('u', 376.5, 202.5),
    ('v', 423.5, 207.5),
    ('w', 472.5, 222.5),
    ('x', 521.5, 244.5),
    ('y', 557.5, 267.5),
    ('z', 601.5, 300.5),
    ('caps', 151.5, 60.5),
    ('1', 141.5, 334.5),
    ('2', 181.5, 336.5),
    ('3', 222.5, 340.5),
    ('4', 267.5, 335.5),
    ('5', 318.5, 342.5),
    ('6', 358.5, 337.5),
    ('7', 400.5, 338.5),
    ('8', 441.5, 338.5),
    ('9', 488.5, 334.5),
    ('0', 535.5, 344.5),
    ('/', 522.5, 63.5),
    ('+', 256.5, 409.5),
    ('+', 266.5, 400.5),
    ('=', 435.5, 406.5),
]

def dist2(p, q):
    px, py = p
    qx, qy = q
    return (px - qx) ** 2 + (py - qy) ** 2

caps = False
def get_text(pos, tick):
    global caps

    mini, minch = None, None
    for ch, x, y in positions:
        dist = dist2(pos, (x, y))
        if mini is None or dist < mini:
            mini = dist
            minch = ch
    
    if mini > 180.0:
        print(f"tick {tick} is wrong: {mini}, {pos}, {minch}")
        return True, None
    elif mini > 120.0:
        print(f"tick {tick} is weird: {mini}, {pos}, {minch}")
        flag = True
    else:
        flag = False

    if minch == 'caps':
        caps = not caps
        return flag, ""

    if caps:
        return flag, minch.lower()
    else:
        return flag, minch.upper()

def get_background(s):
    output = np.median(s, axis=0).astype(np.uint8)
    return output

def stationary(s):
    bg = get_background(s)
    temp = s - bg
    return False

def compute_flow(prev_frame, next_frame, prev_points):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_points, None)
    return next_points, status

def visualize_flow(frame, prev_points, next_points, status):
    mask = np.zeros_like(frame)
    good_points = prev_points[status == 1]
    for i, (prev, next) in enumerate(zip(prev_points, next_points)):
        x_prev, y_prev = prev.ravel()
        x_next, y_next = next.ravel()
        if status[i] == 1:
            mask = cv2.line(mask, tuple(np.int32([x_next, y_next])), tuple(np.int32([x_prev, y_prev])), (0, 255, 0), 2)
            frame = cv2.circle(frame, tuple(np.int32([x_next, y_next])), 5, (0, 0, 255), -1)
    output = cv2.add(frame, mask)
    cv2.imshow("Sparse Optical Flow", output)
    cv2.waitKey(1)  # adjust the waitKey value as needed

def subtract_and_clamp(image1, image2):
    # Perform subtraction
    result = image1.astype(np.int16) - image2.astype(np.int16)
    # Clamp pixel values to ensure they stay within [0, 255]
    result_clamped = np.clip(result, 0, 255).astype(np.uint8)
    return result_clamped

def compute_centroid(img):
    img = np.array(img)
    y_coords, x_coords = np.indices(img.shape)
    total_weight = np.sum(img)
    centroid_x = np.sum(x_coords * img) / total_weight
    centroid_y = np.sum(y_coords * img) / total_weight
    return centroid_x, centroid_y

f = open('output.txt', 'w')
result = open("flow_results.txt", "w")

N = 100
F = 10
sliding_window = np.zeros((N, 434, 653, 3), dtype=np.uint8)
sliding_idx = 0
flow_window = np.zeros((F, 434, 653, 2))
flow_idx = 0

# random_indices = set()
# for i in range(N):
#     random_indices.add(random.randint(60000, 140000))
# bg = get_background(sliding_window)
# for tick in trange(length):
#     success, frame = video_cap.read()
#     frame = frame[116:580, 300:953]
#     if tick in random_indices:
#         sliding_window[sliding_idx] = frame
#         sliding_idx += 1
#         if sliding_idx == N:
#             break
# bg = get_background(sliding_window)
# cv2.imwrite("bg.png", bg)
# print("DONE BG")
# exit()

bg = cv2.imread("bg.png")

prev_frame = None

for tick in trange(length):
    # print(tick, last_pos)
    success, frame = video_cap.read()
    frame = frame[116:550, 300:953]
    frame = subtract_and_clamp(frame, bg)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if tick > 0:
    #     flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame,
    #                                        None, 
    #                                        0.5, 1, 50, 3, 7, 1.5, 0) 
    #     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    #     mask[..., 0] = angle * 180 / np.pi / 2
    #     mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
    #     rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
    #     # cv2.imshow("dense optical flow", rgb) 
    # else:
    #     mask = np.zeros_like(frame) 
    #     mask[..., 1] = 255

    cv2.imshow("Frame", gray_frame)

    prev_frame = gray_frame

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break

    if tick <= 8185:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_blurred = cv2.blur(gray, (3, 3)) 
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                param2 = 30, minRadius = 30, maxRadius = 55) 

    if detected_circles is None:
        continue

    detected_circles = detected_circles[0, :]
    if last_pos is None:
        assert len(detected_circles) == 1
        x, y, r = detected_circles[0]
        last_pos = (x, y)
        continue

    cl, next_pos = None, None
    for x, y, r in detected_circles:
        if x <= 65.0 and y <= 60.0: # Sun
            continue
        if x >= 580.0 and y <= 75.0: # Moon
            continue

        dist = dist2((x, y), last_pos)

        # cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

        if cl is None or cl > dist:
            cl = dist
            next_pos = x, y
    
    if next_pos is not None:
        result.write(str((tick, int(next_pos[0]), int(next_pos[1]), 0, 0)) + "\n")
        dist = dist2(next_pos, last_pos)
        if dist <= 90.0:
            count += 1
            # if count == 5:
            #     cv2.imwrite(f"frames/frame_{tick}.jpg", frame)
            if count == 10:
                flag, ch = get_text(last_pos, tick)
                if flag:
                    # print (f"Tick {tick} is something wrong, {last_pos}")
                    # cv2.imwrite(f"frames/frame_{tick}.jpg", frame)
                    pass
                else:
                    f.write(ch)
                    f.flush()
        else:
            # if count == 5:
            #     print(f"check frame_{tick - 1}")
            count = 0
        last_pos = next_pos

f.close()
result.close()