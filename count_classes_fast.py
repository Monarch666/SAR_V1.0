import os
import glob

def count_labels_fast():
    class_names = {
        0: "victim_person",
        1: "rescue_personnel",
        2: "thermal_human",
        3: "fire",
        4: "smoke",
        5: "rescue_vehicle",
        6: "structural_damage"
    }

    # Step 1: Walk the E:\Dataset\normalized directory to find all label files
    print("Pre-indexing label files on disk...")
    label_files_set = set()
    normalized_dir = r"E:\Dataset\normalized"
    
    # Walk all dataset folders
    for root, dirs, files in os.walk(normalized_dir):
        if "labels" in root:
            for file in files:
                if file.endswith(".txt"):
                    # Store absolute path in lowercase for easy match
                    abs_path = os.path.abspath(os.path.join(root, file)).lower()
                    label_files_set.add(abs_path)
                    
    print(f"Indexed {len(label_files_set)} label files successfully.")

    def analyze_split(txt_path):
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
                
            # Convert image path to label path
            if "\\images\\" in img_path:
                head, tail = img_path.rsplit("\\images\\", 1)
                label_path = os.path.join(head, "labels", tail)
            elif "/images/" in img_path:
                head, tail = img_path.rsplit("/images/", 1)
                label_path = os.path.join(head, "labels", tail)
            else:
                label_path = img_path.replace("images", "labels")
                
            ext = os.path.splitext(label_path)[1]
            label_path = label_path.replace(ext, ".txt")
            abs_label_path = os.path.abspath(label_path).lower()
            
            # Check set membership (O(1) lookup!)
            if abs_label_path in label_files_set:
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
        for cid in sorted(class_names.keys()):
            count = class_counts.get(cid, 0)
            name = class_names[cid]
            percentage = (count / total_labels * 100) if total_labels > 0 else 0
            print(f"    Class {cid} ({name}): {count} instances ({percentage:.2f}%)")

    analyze_split(r"E:\Dataset\merged_final\train.txt")
    analyze_split(r"E:\Dataset\merged_final\val.txt")
    analyze_split(r"E:\Dataset\merged_final\test.txt")

if __name__ == "__main__":
    count_labels_fast()
