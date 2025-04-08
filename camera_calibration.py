import numpy as np
import cv2 as cv
import glob
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont

def put_text_on_image(img, text, position, font_size=20, font_color=(0, 255, 0), font_path=None):
    """
    한글을 포함한 텍스트를 이미지에 표시
    
    Args:
        img: OpenCV BGR 이미지
        text: 표시할 텍스트 (한글 가능)
        position: 텍스트 표시 위치 (x, y)
        font_size: 폰트 크기
        font_color: 폰트 색상 (B, G, R)
        font_path: 폰트 파일 경로 (None이면 기본 폰트 사용)
    
    Returns:
        한글 텍스트가 추가된 이미지
    """
    # RGB로 변환 (PIL은 RGB 형식 사용)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 폰트 설정
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Windows 시스템에서 기본 한글 폰트
            # macOS, Linux 등에서는 다른 기본 폰트 사용 가능
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("malgun.ttf", font_size)
            elif os.name == 'posix':  # macOS/Linux
                # macOS와 Linux의 일반적인 폰트 경로
                font_paths = [
                    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux (Ubuntu)
                    "/usr/share/fonts/nanum/NanumGothic.ttf"  # 다른 Linux 배포판
                ]
                
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, font_size)
                        break
                else:  # 모든 경로에서 폰트를 찾지 못한 경우
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
    except:
        # 폰트를 찾을 수 없으면 기본 폰트 사용
        font = ImageFont.load_default()
    
    # RGB 색상을 PIL 형식으로 변환 (BGR에서 RGB로)
    color_rgb = (font_color[2], font_color[1], font_color[0])
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color_rgb)
    
    # PIL 이미지를 OpenCV 형식으로 변환
    img_with_text = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    
    return img_with_text

def select_images_gui():
    """
    GUI를 사용하여 이미지 파일 선택
    
    Returns:
        선택된 이미지 파일 경로 리스트
    """
    # tkinter 루트 창 생성
    root = tk.Tk()
    root.withdraw()  # 루트 창 숨기기
    
    # 파일 선택 대화상자 표시
    file_paths = filedialog.askopenfilenames(
        title="캘리브레이션할 이미지 선택",
        filetypes=[
            ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
            ("모든 파일", "*.*")
        ]
    )
    
    # 선택된 파일 경로 리스트 반환
    return list(file_paths)

def select_video_gui():
    """
    GUI를 사용하여 비디오 파일 선택
    
    Returns:
        선택된 비디오 파일 경로
    """
    # tkinter 루트 창 생성
    root = tk.Tk()
    root.withdraw()  # 루트 창 숨기기
    
    # 파일 선택 대화상자 표시
    file_path = filedialog.askopenfilename(
        title="캘리브레이션할 비디오 선택",
        filetypes=[
            ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
            ("모든 파일", "*.*")
        ]
    )
    
    return file_path

def select_directory_gui():
    """
    GUI를 사용하여 이미지가 있는 디렉토리 선택
    
    Returns:
        선택된 디렉토리 경로
    """
    # tkinter 루트 창 생성
    root = tk.Tk()
    root.withdraw()  # 루트 창 숨기기
    
    # 디렉토리 선택 대화상자 표시
    directory = filedialog.askdirectory(
        title="이미지가 있는 디렉토리 선택"
    )
    
    return directory

def select_img_from_video(video_path, board_pattern, wait_msec=30):
    """
    비디오에서 체스보드 패턴이 있는 이미지를 선택하는 GUI 도구
    
    Args:
        video_path: 비디오 파일 경로 또는 카메라 인덱스
        board_pattern: 체스보드 패턴 (가로, 세로) 내부 코너 수
        wait_msec: 화면 대기 시간 (밀리초)
    
    Returns:
        선택된, 이미지 리스트
    """
    # 비디오 열기
    if isinstance(video_path, str) and video_path.isdigit():
        video_path = int(video_path)  # 카메라 인덱스인 경우
        
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print(f"비디오를 열 수 없음: {video_path}")
        return []

    # 비디오 정보 가져오기
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print(f"비디오 정보: {width}x{height}, {fps}fps, 총 {total_frames}프레임")
    
    # 창 생성
    window_name = 'Camera Calibration - Select Images'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, width, height)
    
    # Select images
    selected_images = []
    frame_count = 0
    paused = False
    show_corners = False
    current_frame = None
    
    # 결과 디렉토리
    results_dir = 'calibration_results'
    frames_dir = os.path.join(results_dir, 'selected_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print("\n===== 비디오에서 이미지 선택 =====")
    print("스페이스바: 일시정지/재생")
    print("일시정지 상태에서:")
    print("  C: 코너 검출 확인")
    print("  엔터: 현재 프레임 선택")
    print("  →: 다음 프레임")
    print("ESC: 종료")
    print("==============================\n")
    
    while True:
        if not paused:
            # 재생 중일 때만 다음 프레임 읽기
            valid, current_frame = video.read()
            if not valid:
                print("비디오의 끝에 도달했습니다.")
                break
            frame_count += 1
        
        if current_frame is None:
            break
        
        # 디스플레이용 이미지 복사
        display = current_frame.copy()
        
        # 추가 정보 표시 (한글 지원)
        progress = frame_count / total_frames * 100 if total_frames > 0 else 0
        display = put_text_on_image(display, f'프레임: {frame_count}/{total_frames} ({progress:.1f}%)', 
                  (10, 25), font_size=20, font_color=(0, 255, 0))
        display = put_text_on_image(display, f'선택된 이미지: {len(selected_images)}개', 
                  (10, 50), font_size=20, font_color=(0, 255, 0))
        
        if paused:
            display = put_text_on_image(display, "일시정지됨", 
                      (width - 150, 25), font_size=20, font_color=(0, 0, 255))
        
        # 코너 표시 (일시정지 상태에서 C 키를 눌렀을 때)
        if show_corners:
            gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            
            if complete:
                # 코너 위치 정제
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                pts = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
                
                # 코너 그리기
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                display = put_text_on_image(display, "체스보드 패턴 발견됨!", 
                          (width//2 - 200, height - 30), font_size=25, font_color=(0, 255, 0))
            else:
                display = put_text_on_image(display, "체스보드 패턴 찾을 수 없음", 
                          (width//2 - 200, height - 30), font_size=25, font_color=(0, 0, 255))
        
        # 이미지 표시
        cv.imshow(window_name, display)
        
        # 키 이벤트 처리
        key = cv.waitKey(1 if paused else wait_msec)
        
        if key == 27:  # ESC: 종료
            break
        elif key == ord(' '):  # 스페이스바: 일시정지/재생 토글
            paused = not paused
            show_corners = False
        elif key == ord('c') and paused:  # C: 코너 표시 (일시정지 상태에서만)
            show_corners = not show_corners
        elif key == 13 and paused:  # 엔터: 현재 프레임 선택 (일시정지 상태에서만)
            # 체스보드 패턴 확인
            gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            
            if complete:
                selected_images.append(current_frame.copy())
                # 선택한 이미지 저장
                frame_path = os.path.join(frames_dir, f'selected_frame_{len(selected_images):03d}.jpg')
                cv.imwrite(frame_path, current_frame)
                print(f"프레임 {frame_count} 선택됨 (총 {len(selected_images)}개)")
            else:
                print(f"프레임 {frame_count}에서 체스보드 패턴을 찾을 수 없습니다.")
        elif key == 83 and paused:  # 오른쪽 화살표: 다음 프레임 (일시정지 상태에서만)
            valid, current_frame = video.read()
            if valid:
                frame_count += 1
            else:
                print("마지막 프레임입니다.")
    
    # 자원 해제
    video.release()
    cv.destroyAllWindows()
    
    # 체스보드 패턴 검사 및 선택된 이미지 없는 경우 자동 모드 제안
    if len(selected_images) == 0:
        print("선택된 이미지가 없습니다.")
        return []
    
    print(f"총 {len(selected_images)}개 이미지 선택됨")
    return selected_images

def calib_camera_from_chessboard(images, board_pattern, board_cellsize):
    """
    체스보드 이미지에서 카메라 캘리브레이션 수행
    
    Args:
        images: 이미지 리스트
        board_pattern: 체스보드 패턴 (가로, 세로) 내부 코너 수
        board_cellsize: 체스보드 셀 크기(m)
    
    Returns:
        rms, K, dist_coeff, rvecs, tvecs
    """
    # 결과 저장용 디렉토리
    results_dir = 'calibration_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 코너 검출을 위한 이미지 포인트와 객체 포인트
    img_points = []
    obj_points = []
    frame_size = None
    
    # 각 이미지에서 코너 검출
    for i, img in enumerate(images):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # 첫 이미지에서 프레임 크기 저장
        if frame_size is None:
            frame_size = gray.shape[::-1]  # (width, height)
        
        # 체스보드 코너 검출
        ret, corners = cv.findChessboardCorners(gray, board_pattern, None)
            
        if ret:
            # 코너 위치 정제
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            img_points.append(corners)
            
            # 체스보드의 3D 포인트 준비
            objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * board_cellsize
            obj_points.append(objp)
            
            print(f"[{i+1}/{len(images)}] 체스보드 코너 검출 성공")
            
            # 코너 검출 결과 시각화
            corners_img = img.copy()
            cv.drawChessboardCorners(corners_img, board_pattern, corners, ret)
            
            # 코너 검출 결과 저장
            corners_dir = os.path.join(results_dir, 'detected_corners')
            os.makedirs(corners_dir, exist_ok=True)
            cv.imwrite(os.path.join(corners_dir, f'corners_{i:03d}.jpg'), corners_img)
        else:
            print(f"[{i+1}/{len(images)}] 체스보드 코너 검출 실패")
    
    if not img_points:
        print("검출된 체스보드 코너가 없습니다. 캘리브레이션을 진행할 수 없습니다.")
        return None, None, None, None, None
    
    print(f"카메라 캘리브레이션 수행 중... (사용 이미지: {len(img_points)}개)")
    
    # 카메라 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, frame_size, None, None)
    
    print(f"캘리브레이션 완료: 재투영 오차 = {ret}")
    
    # 재투영 오차 계산
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(obj_points)
    print(f"평균 재투영 오차: {mean_error}")
    
    return ret, mtx, dist, rvecs, tvecs

def save_calibration_results(K, dist_coeff, board_pattern, board_cellsize, rms, img_count):
    """
    캘리브레이션 결과를 파일로 저장
    
    Args:
        K: 카메라 행렬
        dist_coeff: 왜곡 계수
        board_pattern: 체스보드 패턴 (가로, 세로) 내부 코너 수
        board_cellsize: 체스보드 셀 크기(m)
        rms: 평균 제곱근 오차
        img_count: 사용된 이미지 수
    """
    # 결과 디렉토리
    results_dir = 'calibration_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 텍스트 파일 생성
    with open(os.path.join(results_dir, 'calibration_results.txt'), 'w') as f:
        f.write("==== 카메라 캘리브레이션 결과 ====\n")
        f.write(f"캘리브레이션 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"체스보드 크기: {board_pattern[0]}x{board_pattern[1]}\n")
        f.write(f"사각형 크기: {board_cellsize * 100} cm\n")
        f.write(f"사용된 이미지 수: {img_count}\n\n")
        
        f.write("카메라 내부 파라미터 (Camera Matrix):\n")
        f.write(f"{K}\n\n")
        
        f.write("왜곡 계수 (Distortion Coefficients):\n")
        f.write(f"{dist_coeff}\n\n")
        
        f.write(f"평균 재투영 오차 (RMSE): {rms}")
    
    # 캘리브레이션 매트릭스 저장 (OpenCV에서 사용하기 위한 형식)
    np.savez(os.path.join(results_dir, 'calibration_data.npz'),
             camera_matrix=K,
             distortion_coeffs=dist_coeff,
             frame_size=(0, 0))  # 프레임 크기도 저장
    
    print(f"캘리브레이션 결과가 {results_dir} 디렉토리에 저장되었습니다.")

def visualize_undistortion(K, dist_coeff, images):
    """
    왜곡 보정 결과 시각화
    
    Args:
        K: 카메라 행렬
        dist_coeff: 왜곡 계수
        images: 이미지 리스트
    """
    # 결과 디렉토리
    results_dir = 'calibration_results'
    undistort_dir = os.path.join(results_dir, 'undistorted_images')
    os.makedirs(undistort_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        
        # 최적의 카메라 매트릭스 계산 (alpha=1)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 1, (w, h))
        
        # 이미지 왜곡 보정
        dst = cv.undistort(img, K, dist_coeff, None, newcameramtx)
        
        # ROI 자르기
        x, y, w, h = roi
        if all([x, y, w, h]):  # ROI가 유효한 경우
            dst = dst[y:y+h, x:x+w]
        
        # 보정된 이미지 저장
        output_path = os.path.join(undistort_dir, f'undistorted_{i:03d}.jpg')
        cv.imwrite(output_path, dst)
        
        print(f"이미지 왜곡 보정 완료: {i+1}/{len(images)}")
    
    print(f"왜곡 보정된 이미지가 {undistort_dir} 디렉토리에 저장되었습니다.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='카메라 캘리브레이션 도구')
    
    # 입력 소스 지정
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--images', nargs='+', help='캘리브레이션할 이미지 파일 경로')
    input_group.add_argument('--image_dir', help='이미지 파일이 있는 디렉토리 경로')
    input_group.add_argument('--video', help='캘리브레이션할 비디오 파일 경로 또는 카메라 인덱스 (예: 0)')
    input_group.add_argument('--gui', action='store_true', help='GUI 모드로 이미지 선택')
    
    # 체스보드 설정
    parser.add_argument('--chessboard_size', nargs=2, type=int, default=[10, 7],
                       help='체스보드 내부 코너 수 (가로, 세로) (기본값: 10 7)')
    parser.add_argument('--square_size', type=float, default=2.5,
                       help='체스보드 사각형의 실제 크기(cm) (기본값: 2.5)')
    
    args = parser.parse_args()
    
    # 체스보드 크기 설정
    board_pattern = tuple(args.chessboard_size)
    board_cellsize = args.square_size / 100.0  # cm -> m 변환
    
    print(f'## 카메라 캘리브레이션 시작')
    print(f'* 체스보드 패턴: {board_pattern[0]}x{board_pattern[1]} (내부 코너)')
    print(f'* 셀 크기: {args.square_size} cm ({board_cellsize*1000:.1f} mm)')
    
    # 이미지 처리
    images = []
    
    if args.gui:
        # GUI를 통한 파일 선택
        print("GUI 모드로 이미지 선택")
        choice = input("이미지 선택 방법을 선택하세요: \n1. 개별 이미지 파일 선택\n2. 이미지 폴더 선택\n3. 비디오 파일 또는 카메라 선택\n>> ")
        
        if choice == '1':
            # 개별 이미지 파일 선택
            file_paths = select_images_gui()
            if not file_paths:
                print("이미지가 선택되지 않았습니다.")
                return
                
            print(f"선택된 이미지 파일: {len(file_paths)}개")
            for img_path in file_paths:
                img = cv.imread(img_path)
                if img is not None:
                    images.append(img)
        
        elif choice == '2':
            # 이미지 폴더 선택
            directory = select_directory_gui()
            if not directory:
                print("폴더가 선택되지 않았습니다.")
                return
                
            print(f"선택된 폴더: {directory}")
            image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            image_paths = []
            for ext in image_types:
                image_paths.extend(glob.glob(os.path.join(directory, ext)))
            
            for img_path in image_paths:
                img = cv.imread(img_path)
                if img is not None:
                    images.append(img)
        
        elif choice == '3':
            # 비디오 파일 선택
            video_choice = input("비디오 소스를 선택하세요: \n1. 비디오 파일\n2. 카메라\n>> ")
            
            if video_choice == '1':
                video_path = select_video_gui()
                if not video_path:
                    print("비디오 파일이 선택되지 않았습니다.")
                    return
                
                print(f"선택된 비디오 파일: {video_path}")
            elif video_choice == '2':
                camera_idx = input("카메라 인덱스를 입력하세요 (기본값: 0): ")
                if not camera_idx:
                    camera_idx = "0"
                
                video_path = int(camera_idx)
                print(f"선택된 카메라 인덱스: {camera_idx}")
            else:
                print("잘못된 선택입니다.")
                return
            
            # 비디오에서 이미지 선택
            images = select_img_from_video(video_path, board_pattern)
        
        else:
            print("잘못된 선택입니다.")
            return
    
    elif args.video:
        # 비디오에서 이미지 선택
        images = select_img_from_video(args.video, board_pattern)
    
    elif args.images:
        # 개별 이미지 파일 처리
        for img_path in args.images:
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
    elif args.image_dir:
        # 디렉토리 내 모든 이미지 처리
        image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in image_types:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        for img_path in image_paths:
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
    else:
        print("이미지 소스를 지정해주세요. (--images, --image_dir, --video 또는 --gui)")
        return
    
    # 충분한 이미지가 처리되었는지 확인
    if len(images) < 5:
        print("캘리브레이션을 위한 충분한 이미지가 없습니다. (최소 5개 필요)")
        return
    
    print(f'* 선택된 이미지 수: {len(images)}')
    
    # 카메라 캘리브레이션 수행
    print('* 카메라 캘리브레이션 수행 중...')
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(
        images, board_pattern, board_cellsize)
    
    if K is None:
        print("캘리브레이션에 실패했습니다.")
        return
    
    # 결과 저장 및 출력
    save_calibration_results(K, dist_coeff, board_pattern, board_cellsize, rms, len(images))
    
    # 왜곡 보정 시각화
    print('* 왜곡 보정 결과 시각화 중...')
    visualize_undistortion(K, dist_coeff, images[:5])  # 처음 5개 이미지만 시각화
    
    # 결과 출력
    print("\n====== 카메라 캘리브레이션 결과 ======")
    print(f"* RMS 오차 = {rms}")
    print(f"* 카메라 행렬 (K) = \n{K}")
    print(f"* 왜곡 계수 (k1, k2, p1, p2, k3) = {dist_coeff.flatten()}")
    print("====================================")

if __name__ == "__main__":
    main() 