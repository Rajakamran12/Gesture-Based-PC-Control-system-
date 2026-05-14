#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick script to check dataset status"""

import os

dataset_dir = 'dataset'
folders = ['right_finger', 'left_finger', 'open_hand', 'close_hand', 'two_finger']

print("="*60)
print("DATASET STATUS CHECK")
print("="*60)

total_images = 0
for folder in folders:
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        count = len(images)
        total_images += count
        status = "✅" if count >= 50 else "⚠️" if count > 0 else "❌"
        print(f"{status} {folder:15s}: {count:4d} images")
    else:
        print(f"❌ {folder:15s}: Folder not found")

print("="*60)
print(f"Total images: {total_images}")
print("="*60)

if total_images < 250:
    print("\n⚠️  WARNING: You need at least 50 images per class (250 total)")
    print("   Recommended: 100-200 images per class for better accuracy")
else:
    print("\n✅ Dataset looks good! Ready to train.")

