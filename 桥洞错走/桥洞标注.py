import cv2
import numpy as np

# 初始化摄像头
cap = cv2.VideoCapture(0)


# 模拟桥洞检测的模板
def detect_bridge_holes(frame):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用Canny边缘检测找到桥洞的边缘
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 找到图像中的所有轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bridge_holes = []
    for contour in contours:
        # 计算每个轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 只选择符合条件的桥洞（如面积大小）
        if w > 50 and h > 50:
            bridge_holes.append((x, y, w, h))

    return bridge_holes


# 模拟船只定位（假设船只是蓝色的）
def detect_ship(frame):
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 定义蓝色的HSV范围
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # 创建掩模来提取蓝色区域
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # 找到船只的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 返回船只的位置
        return x + w // 2, y + h // 2  # 返回船只的中心点坐标
    return None


# 主循环，处理每一帧
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测桥洞并标注
    bridge_holes = detect_bridge_holes(frame)
    for (x, y, w, h) in bridge_holes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 检测船只并标注
    ship_position = detect_ship(frame)
    if ship_position:
        cx, cy = ship_position
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # 判断船只是否在错误的桥洞区域内（简化示例：假设桥洞的左边界为x坐标）
        for (x, y, w, h) in bridge_holes:
            if x < cx < x + w and y < cy < y + h:
                print("船只在正确的桥洞区域内")
                break
        else:
            print("船只进入了错误的桥洞区域，触发报警！")

    # 显示图像
    cv2.imshow('Bridge Hole Detection', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
