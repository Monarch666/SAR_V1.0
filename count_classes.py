import os

def count_labels(txt_path):
    print(f"\nAnalyzing dataset split: {txt_path}")
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return
        
    class_counts = {}
    total_images_with_labels = 0
    total_images_without_labels = 0
    total_labels = 0
    
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        image_paths = f.read().strip().split('\n')
        
    for img_path in image_paths:
        img_path = img_path.strip()
        if not img_path:
            continue
            
        # Replace only the directory portion "\images\" or "/images/" with "\labels\" or "/labels/"
        # Find the last occurrence of "/images/" or "\images\" to change the folder, not the filename
        if "\\images\\" in img_path:
            # Split and replace folder
            head, tail = img_path.rsplit("\\images\\", 1)
            label_path = os.path.join(head, "labels", tail)
        elif "/images/" in img_path:
            head, tail = img_path.rsplit("/images/", 1)
            label_path = os.path.join(head, "labels", tail)
        else:
            # Fallback
            label_path = img_path.replace("images", "labels")
            
        ext = os.path.splitext(label_path)[1]
        label_path = label_path.replace(ext, ".txt")
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8', errors='ignore') as lf:
                    content = lf.read().strip()
                    lines = content.split('\n')
                    lines = [l for l in lines if l.strip()]
                    if lines:
                        total_images_with_labels += 1
                        for line in lines:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                total_labels += 1
                    else:
                        total_images_without_labels += 1
            except Exception as e:
                total_images_without_labels += 1
        else:
            total_images_without_labels += 1
            
    print(f"  Total Images: {len(image_paths)}")
    print(f"  Images with labels: {total_images_with_labels}")
    print(f"  Background images (no labels): {total_images_without_labels}")
    print(f"  Total label instances: {total_labels}")
    print("  Class distribution:")
    class_names = {
        0: "victim_person",
        1: "rescue_personnel",
        2: "thermal_human",
        3: "fire",
        4: "smoke",
        5: "rescue_vehicle",
        6: "structural_damage"
    }
    for cid in sorted(class_names.keys()):
        count = class_counts.get(cid, 0)
        name = class_names[cid]
        percentage = (count / total_labels * 100) if total_labels > 0 else 0
        print(f"    Class {cid} ({name}): {count} instances ({percentage:.2f}%)")

count_labels(r"E:\Dataset\merged_final\train.txt")
count_labels(r"E:\Dataset\merged_final\val.txt")
count_labels(r"E:\Dataset\merged_final\test.txt")
