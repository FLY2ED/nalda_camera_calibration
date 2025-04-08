import numpy as np
import cv2 as cv
import os
import glob
import argparse
import matplotlib
# GUI 사용하지 않는 백엔드로 설정 (NumPy 2.0 호환성 문제 해결)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time

class DistortionCorrection:
    """
    카메라 캘리브레이션 결과를 이용하여 렌즈 왜곡을 보정하는 클래스
    """
    def __init__(self, config_file=None):
        """
        왜곡 보정을 위한 클래스
        
        Args:
            config_file: 캘리브레이션 데이터가 포함된 .npz 파일
        """
        self.K = None
        self.dist_coeff = None
        self.balance = 0.0  # 초기 balance 값
        
        # 결과 디렉토리
        self.output_dir = 'distortion_correction_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        if config_file is not None:
            self.load_calibration(config_file)
    
    def load_calibration(self, calibration_file):
        """
        캘리브레이션 결과 파일 로드
        
        Args:
            calibration_file: 캘리브레이션 결과 파일 경로 (.npz)
            
        Returns:
            로드 성공 여부
        """
        try:
            if calibration_file.endswith('.npz'):
                # .npz 파일에서 로드
                data = np.load(calibration_file)
                self.K = data['camera_matrix']
                self.dist_coeff = data['distortion_coeffs']
            else:
                print(f"지원되지 않는 캘리브레이션 파일 형식: {calibration_file}")
                return False
                
            print(f"캘리브레이션 데이터 로드 성공: {calibration_file}")
            print(f"카메라 매트릭스:\n{self.K}")
            print(f"왜곡 계수:\n{self.dist_coeff}")
            
            return True
        except Exception as e:
            print(f"캘리브레이션 파일 로드 중 오류 발생: {e}")
            return False
    
    def correct_image(self, img, balance=None):
        """
        이미지 왜곡 보정
        
        Args:
            img: 보정할 이미지
            balance: 왜곡 보정 balance (None이면 현재 설정 사용)
            
        Returns:
            corrected_img: 왜곡 보정된 이미지
        """
        assert self.K is not None, "캘리브레이션 데이터 필요"
        
        if balance is not None:
            self.balance = balance
        
        h, w = img.shape[:2]
        img_size = (w, h)
        
        # camera_calibration.py와 동일한 방식으로 undistort 수행
        # getOptimalNewCameraMatrix 호출
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            self.K, self.dist_coeff, img_size, self.balance, img_size)
        
        # undistort 직접 호출 (매핑 테이블 사용하지 않음)
        corrected_img = cv.undistort(img, self.K, self.dist_coeff, None, newcameramtx)
        
        # ROI 적용 (옵션)
        if roi[2] > 0 and roi[3] > 0 and self.balance > 0:
            x, y, w, h = roi
            corrected_img = corrected_img[y:y+h, x:x+w]
        
        return corrected_img
    
    def correct_and_save_image(self, image_path, balance=1.0, crop=True):
        """
        이미지 파일을 로드하여 왜곡 보정 후 저장
        
        Args:
            image_path: 보정할 이미지 파일 경로
            balance: 왜곡 보정의 강도를 조절하는 파라미터 (0.0~1.0)
            crop: 보정 후 이미지 자르기 여부
            
        Returns:
            성공 여부, 원본 이미지, 보정된 이미지
        """
        # 이미지 로드
        image = cv.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없음: {image_path}")
            return False, None, None
        
        # 이미지 왜곡 보정
        undistorted_img = self.correct_image(image, balance)
        
        # 결과 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f'undistorted_{base_name}.jpg')
        cv.imwrite(output_path, undistorted_img)
        
        # 비교 이미지 생성
        # 두 이미지의 크기를 맞추기
        orig_h, orig_w = image.shape[:2]
        undist_h, undist_w = undistorted_img.shape[:2]
        
        # 더 작은 크기에 맞추기
        target_w = min(orig_w, undist_w)
        target_h = min(orig_h, undist_h)
        
        # 이미지 리사이즈
        image_resized = cv.resize(image, (target_w, target_h))
        undist_resized = cv.resize(undistorted_img, (target_w, target_h))
        
        # 이미지 가로로 합치기
        comparison = np.hstack((image_resized, undist_resized))
        
        # 비교 이미지에 텍스트 추가
        cv.putText(comparison, 'Original', (50, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(comparison, 'Undistorted', (target_w + 50, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 비교 이미지 저장
        comparison_path = os.path.join(self.output_dir, f'comparison_{base_name}.jpg')
        cv.imwrite(comparison_path, comparison)
        
        print(f"왜곡 보정 완료: {output_path}")
        print(f"비교 이미지 저장: {comparison_path}")
        
        return True, image, undistorted_img
    
    def process_batch(self, image_paths, balance=1.0, crop=True):
        """
        여러 이미지 파일들을 일괄 처리
        
        Args:
            image_paths: 처리할 이미지 파일 경로 리스트
            balance: 왜곡 보정 파라미터 (0.0~1.0)
            crop: 이미지 자르기 여부
            
        Returns:
            성공적으로 처리된 이미지 수
        """
        successful_count = 0
        
        for i, image_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] 이미지 처리 중: {image_path}")
            success, _, _ = self.correct_and_save_image(image_path, balance, crop)
            if success:
                successful_count += 1
        
        print(f"총 {len(image_paths)}개 이미지 중 {successful_count}개 성공적으로 처리됨")
        return successful_count
    
    def process_video(self, in_video_path, out_video_path, balance=None, progress_callback=None):
        """
        비디오 파일 왜곡 보정 처리
        
        Args:
            in_video_path: 입력 비디오 경로
            out_video_path: 출력 비디오 경로
            balance: 왜곡 보정 balance
            progress_callback: 진행 상황 콜백 함수
        
        Returns:
            처리 결과 (성공: True, 실패: False)
        """
        assert self.K is not None, "캘리브레이션 데이터 필요"
        
        if balance is not None:
            self.balance = balance
        
        # 입력 비디오 열기
        cap = cv.VideoCapture(in_video_path)
        if not cap.isOpened():
            print(f"에러: 비디오 파일을 열 수 없음 - {in_video_path}")
            return False
        
        # 비디오 속성 가져오기
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # 최적의 카메라 매트릭스 미리 계산
        img_size = (width, height)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            self.K, self.dist_coeff, img_size, self.balance, img_size)
        
        # 출력 비디오 설정
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID'
        out = cv.VideoWriter(out_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"에러: 출력 비디오 파일을 생성할 수 없음 - {out_video_path}")
            cap.release()
            return False
        
        # 프레임별 처리
        frame_count = 0
        processing_start = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 왜곡 보정 (camera_calibration.py와 같은 방식 사용)
                corrected = cv.undistort(frame, self.K, self.dist_coeff, None, newcameramtx)
                
                # ROI 적용 (옵션)
                if roi[2] > 0 and roi[3] > 0 and self.balance > 0:
                    x, y, w, h = roi
                    # 출력 크기 유지를 위해 ROI 자르기 대신 원본 크기의 이미지 생성
                    # 주의: 이 부분은 출력 비디오 크기를 유지하기 위함임
                    # 실제로 자르고 싶으면 아래 주석을 해제하고 출력 비디오 크기도 조정해야 함
                    # corrected = corrected[y:y+h, x:x+w]
                
                # 출력 비디오에 저장
                out.write(corrected)
                
                # 진행 상황 업데이트
                frame_count += 1
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress)
                
                # 1000 프레임마다 처리 속도 출력
                if frame_count % 1000 == 0:
                    elapsed = time.time() - processing_start
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    print(f"처리 중... {frame_count}/{total_frames} 프레임 "
                          f"({progress*100:.1f}%, {fps_processing:.1f} fps)")
        
        except Exception as e:
            print(f"비디오 처리 중 에러 발생: {str(e)}")
            cap.release()
            out.release()
            return False
        
        # 비디오 종료
        cap.release()
        out.release()
        
        processing_time = time.time() - processing_start
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        print(f"비디오 처리 완료: {frame_count} 프레임, {processing_time:.1f}초 "
              f"(평균 {avg_fps:.1f} fps)")
        
        return True
    
    def create_before_after_animation(self, image_path, balance=1.0, crop=True, duration=5.0):
        """
        원본과 보정 이미지를 번갈아가며 보여주는 애니메이션 생성
        
        Args:
            image_path: 보정할 이미지 파일 경로
            balance: 왜곡 보정 파라미터 (0.0~1.0)
            crop: 이미지 자르기 여부
            duration: 애니메이션 길이 (초)
            
        Returns:
            성공 여부
        """
        # 이미지 로드
        image = cv.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없음: {image_path}")
            return False
        
        # 이미지 왜곡 보정
        undistorted_img = self.correct_image(image, balance)
        
        # RGB로 변환
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        undistorted_rgb = cv.cvtColor(undistorted_img, cv.COLOR_BGR2RGB)
        
        # 애니메이션 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 이미지 최초 표시
        img_obj = ax.imshow(image_rgb)
        title_obj = ax.set_title('Original Image', fontsize=15)
        
        # 애니메이션 프레임 함수
        def update(frame):
            if frame % 2 == 0:
                img_obj.set_array(image_rgb)
                title_obj.set_text('Original Image')
            else:
                img_obj.set_array(undistorted_rgb)
                title_obj.set_text('Undistorted Image')
            return img_obj, title_obj
        
        # 1초에 2프레임 (0.5초마다 전환)
        fps = 2
        frames = int(duration * fps)
        
        # 애니메이션 생성
        ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
        
        # GIF 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f'animation_{base_name}.gif')
        
        # 애니메이션 저장
        ani.save(output_path, writer='pillow', fps=fps)
        
        plt.close(fig)
        print(f"애니메이션 생성 완료: {output_path}")
        return True
    
    def create_balance_comparison(self, image_path, crop=True):
        """
        다양한 balance 값에 대한 비교 이미지 생성
        
        Args:
            image_path: 보정할 이미지 파일 경로
            crop: 이미지 자르기 여부
            
        Returns:
            성공 여부
        """
        # 이미지 로드
        image = cv.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없음: {image_path}")
            return False
        
        # balance 값 목록
        balance_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        undistorted_imgs = []
        
        # 각 balance 값에 대해 왜곡 보정
        for balance in balance_values:
            undistorted_img = self.correct_image(image, balance)
            undistorted_imgs.append(undistorted_img)
        
        # RGB로 변환
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        undistorted_rgbs = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in undistorted_imgs]
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        # 원본 이미지
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # 다양한 balance 값에 대한 왜곡 보정 이미지
        for i, (balance, undist_rgb) in enumerate(zip(balance_values, undistorted_rgbs)):
            axes[i+1].imshow(undist_rgb)
            axes[i+1].set_title(f'balance={balance}', fontsize=12)
            axes[i+1].axis('off')
        
        plt.tight_layout()
        
        # 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f'balance_comparison_{base_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        print(f"balance 비교 이미지 생성 완료: {output_path}")
        return True

def main():
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='렌즈 왜곡 보정 도구')
    
    # 입력 파일 지정
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', help='처리할 이미지 파일 경로')
    input_group.add_argument('--images', nargs='+', help='처리할 이미지 파일 경로들')
    input_group.add_argument('--image_dir', help='이미지 파일이 있는 디렉토리 경로')
    input_group.add_argument('--video', help='처리할 비디오 파일 경로')
    
    # 캘리브레이션 파일
    parser.add_argument('--calibration', required=True,
                        help='카메라 캘리브레이션 결과 파일 경로 (.npz)')
    
    # 왜곡 보정 옵션
    parser.add_argument('--balance', type=float, default=1.0,
                        help='왜곡 보정 강도 (0.0~1.0, 기본값: 1.0)')
    parser.add_argument('--no_crop', action='store_true',
                        help='왜곡 보정 후 이미지 자르기 비활성화')
    
    # 비디오 처리 옵션
    parser.add_argument('--output_fps', type=float,
                        help='출력 비디오의 FPS (지정하지 않으면 원본 FPS 유지)')
    parser.add_argument('--output_video', 
                        help='출력 비디오 파일 경로 (지정하지 않으면 자동 생성)')
    
    # 애니메이션 생성 옵션
    parser.add_argument('--create_animation', action='store_true',
                        help='원본과 보정 이미지 간의 전환 애니메이션 생성')
    parser.add_argument('--balance_comparison', action='store_true',
                        help='다양한 balance 값에 대한 비교 이미지 생성')
    
    # 파싱
    args = parser.parse_args()
    
    # 왜곡 보정기 생성
    corrector = DistortionCorrection(args.calibration)
    
    # 자르기 옵션
    crop = not args.no_crop
    
    # 입력에 따라 처리
    if args.image:
        # 단일 이미지 처리
        success, _, _ = corrector.correct_and_save_image(args.image, args.balance, crop)
        
        # 애니메이션 생성 (요청된 경우)
        if success and args.create_animation:
            corrector.create_before_after_animation(args.image, args.balance, crop)
        
        # balance 비교 이미지 생성 (요청된 경우)
        if success and args.balance_comparison:
            corrector.create_balance_comparison(args.image, crop)
            
    elif args.images:
        # 여러 이미지 처리
        corrector.process_batch(args.images, args.balance, crop)
        
        # 첫 번째 이미지로 애니메이션/비교 이미지 생성 (요청된 경우)
        if args.create_animation or args.balance_comparison:
            first_image = args.images[0]
            if args.create_animation:
                corrector.create_before_after_animation(first_image, args.balance, crop)
            if args.balance_comparison:
                corrector.create_balance_comparison(first_image, crop)
                
    elif args.image_dir:
        # 디렉토리 내 모든 이미지 처리
        image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in image_types:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        if not image_paths:
            print(f"지정한 디렉토리에 이미지 파일이 없습니다: {args.image_dir}")
            return
        
        corrector.process_batch(image_paths, args.balance, crop)
        
        # 첫 번째 이미지로 애니메이션/비교 이미지 생성 (요청된 경우)
        if args.create_animation or args.balance_comparison:
            first_image = image_paths[0]
            if args.create_animation:
                corrector.create_before_after_animation(first_image, args.balance, crop)
            if args.balance_comparison:
                corrector.create_balance_comparison(first_image, crop)
                
    elif args.video:
        # 비디오 처리
        # 출력 비디오 경로 생성
        if args.output_video:
            output_video_path = args.output_video
        else:
            video_name = os.path.splitext(os.path.basename(args.video))[0]
            output_video_path = os.path.join(corrector.output_dir, f'undistorted_{video_name}.mp4')
        
        # 비디오 왜곡 보정 처리
        corrector.process_video(args.video, output_video_path, args.balance, None)
        print(f"처리된 비디오 저장됨: {output_video_path}")

if __name__ == "__main__":
    main()