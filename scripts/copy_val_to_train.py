#!/usr/bin/env python3
"""
Validation set을 Train set으로 복사하면서 폴더 번호를 재지정하는 스크립트
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # 경로 설정
    val_dir = Path("/home/user/Carla-OpenLane/Carla/Carla-OpenLane/subset_A/val")
    train_dir = Path("/home/user/Carla-OpenLane/Carla/Carla-OpenLane/subset_A/train")

    # validation 폴더 목록 가져오기
    val_folders = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"📁 Validation set에서 {len(val_folders)}개 폴더 발견")

    # train 폴더에서 마지막 번호 찾기
    train_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    last_train_num = int(train_folders[-1].name) if train_folders else 0

    print(f"🔢 Train set의 마지막 번호: {last_train_num:04d}")
    print(f"🔢 새로운 번호 시작: {last_train_num + 1:04d}")

    # 복사할 폴더 개수 (82개 또는 가능한 만큼)
    target_count = min(82, len(val_folders))
    print(f"📋 복사할 폴더 개수: {target_count}개")

    # 복사 시작
    start_num = last_train_num + 1
    copied_count = 0

    print(f"\n🚀 복사 시작...")
    for i, val_folder in enumerate(tqdm(val_folders[:target_count], desc="Copying folders")):
        new_num = start_num + i
        new_folder_name = f"{new_num:04d}"
        new_folder_path = train_dir / new_folder_name

        # 대상 폴더가 이미 존재하는지 확인
        if new_folder_path.exists():
            print(f"⚠️  {new_folder_name} 이미 존재 - 건너뛰기")
            continue

        # 폴더 복사
        try:
            shutil.copytree(val_folder, new_folder_path)
            copied_count += 1
            # print(f"✅ {val_folder.name} → {new_folder_name}")
        except Exception as e:
            print(f"❌ {val_folder.name} 복사 실패: {e}")

    print(f"\n✅ 복사 완료!")
    print(f"📊 복사된 폴더: {copied_count}개")
    print(f"📊 Train set 총 폴더 수: {len(list(train_dir.iterdir()))}개")

    # 복사 결과 확인
    print(f"\n🔍 새로 추가된 폴더 확인:")
    new_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    added_folders = new_folders[-(copied_count):]
    print(f"   처음 5개: {added_folders[:5]}")
    print(f"   마지막 5개: {added_folders[-5:]}")

if __name__ == "__main__":
    main()