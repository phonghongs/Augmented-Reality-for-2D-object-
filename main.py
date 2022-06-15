import numpy as np
import cv2
import time
import threading

HIGH = 480
WIDTH = 640

global imgsrc, obj_stt, isdone
imgcache = imgsrc = cv2.imread('LyHongPhong_AVA.jpg')
obj_stt = 1
isdone = False

def nothing(x):
    pass

# Creating a window with black image
cv2.namedWindow('Result')
# creating trackbars for red color change
cv2.createTrackbar('ratio', 'Result', 1, 100, nothing)

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def get_object_coordinates(image, scale = 0.03):
    h_ob, w_ob, _ = image.shape
    ratio = w_ob / h_ob
    object_y = 1 * scale
    object_z = (1 / ratio) * scale
    cube_coordinates = np.float32([
                            [0, object_y, 0], [0, -object_y, 0], 
                            [0, -object_y, object_z], [0, object_y, object_z]
                        ])
    return cube_coordinates


def warpAndmask(background, imgSrc, coord):
    h_bg, w_bg, _ = background.shape
    (srcH, srcW) = imgSrc.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    (H, _) = cv2.findHomography(srcMat, coord)
    warped = cv2.warpPerspective(imgSrc, H, (w_bg, h_bg))
    # mask = np.zeros((h_bg, w_bg), dtype="uint8")
    # cv2.fillConvexPoly(mask, coord.astype("int32"), (255, 255, 255),
    #     cv2.LINE_AA)

    # rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask = cv2.erode(mask, rect, iterations=2)

    blankIMG = np.zeros((warped.shape), dtype="uint8")

    mask = cv2.inRange(warped, (1, 1, 1), (255, 150, 255))
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(blankIMG.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    # rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask = cv2.erode(mask, rect, iterations=2)

    return warped, mask


def addResult(background, warped, mask):
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(background.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    return output


def objectReply():
    global imgsrc, obj_stt, isdone

    while not isdone:
        obj_1 = cv2.VideoCapture('Video/dragon_1.mp4')
        obj_2 = cv2.VideoCapture('Video/tiger.mp4')
        obj_3 = cv2.VideoCapture('Video/youtube.mp4')
        while not isdone:
            ret = 0
            img = []
            if obj_stt == 1:
                ret, img = obj_1.read()
            elif obj_stt == 2:
                ret, img = obj_2.read()
            elif obj_stt == 3:
                ret, img = obj_3.read()
            
            if ret:
                imgsrc = cv2.flip(img, -1)
                time.sleep(0.03)
            else:
                break

        obj_1.release()
        obj_2.release()
        obj_3.release()


def main():
    global imgsrc, obj_stt, isdone

    mtx, dist = load_coefficients('calibration_chessboard_96deg.yml')
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # detector parameters can be set here (List of detection parameters[3])
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    x = threading.Thread(target=objectReply, args=())
    x.start()

    while not isdone:
        cap = cv2.VideoCapture('Video/video_2.mp4')
        while not isdone:
            val, curr_frame = cap.read()
            imgsrc_cp = imgsrc.copy()
            if val is None or val is False:
                print(" The video has been processed !!")
                break

            rt = cv2.getTrackbarPos('ratio', 'Result') / 100
            if rt == 0:
                rt = 0.01
            curr_frame = cv2.resize(curr_frame, (640, 480))
            curr_frame_orginal = curr_frame.copy()
            gray = cv2.cvtColor(curr_frame_orginal, cv2.COLOR_BGR2GRAY)
            # Detect ArucoMarkers
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if np.all(ids != None):
                cv2.aruco.drawDetectedMarkers(curr_frame_orginal, corners)
                rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

                object_coordinates = get_object_coordinates(imgsrc_cp, scale=rt)
                coordinates_3D, jacobian = cv2.projectPoints(object_coordinates, rvec,
                                                            tvec, mtx, dist)
                coordinates_3D = np.int32(coordinates_3D).reshape(4, 1, 2)

                # for i in range(0, ids.size):
                # # draw axis for the aruco markers
                #     cv2.aruco.drawAxis(curr_frame_orginal, mtx, dist, rvec[i], tvec[i], 0.1)

                warped, mask = warpAndmask(curr_frame_orginal, imgsrc_cp, coordinates_3D)
                result = addResult(curr_frame_orginal, warped, mask)

                cv2.imshow("Result", result)

        ##==========================================================================================
            key = cv2.waitKey(1)
            if key == ord('q'):
                isdone = True
                x.join()
                break
            elif key == ord('1'):
                obj_stt = 1
            elif key == ord('2'):
                obj_stt = 2
            elif key == ord('3'):
                obj_stt = 3
        cap.release()

if __name__ == "__main__":
    main()
    isdone = True
    print("END")
