import cv2

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không mở được webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Không đọc được frame")
            break
        
        cv2.imshow("Phát hiện ngủ khi nhắm mắt", frame)
        
        #Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()