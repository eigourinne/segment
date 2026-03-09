import os
import re

# 定义基础目录和子目录
base_dir = "./data_wound_seg"
sub_dirs = ["val_images", "val_masks"]

# 新文件名起始编号
start_num = 1

# 第一步：分别收集两个目录中符合范围要求的文件，并按原始数字排序
files_to_rename = {}
for sub_dir in sub_dirs:
    dir_path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir_path):
        print(f"警告：目录 {dir_path} 不存在，已跳过。")
        continue

    file_list = []
    for filename in os.listdir(dir_path):
        # 使用正则表达式匹配 fusc_XXXX.png 格式，并提取数字
        match = re.match(r"fusc_(\d{4})\.png", filename)
        if match:
            file_num = int(match.group(1))
            # 检查编号是否在指定范围内 [551, 1010]
            if 551 <= file_num <= 1010:
                file_list.append((file_num, filename))

    # 按提取出的数字排序
    file_list.sort(key=lambda x: x[0])
    files_to_rename[sub_dir] = [fname for _, fname in file_list]
    print(f"在 '{sub_dir}' 中找到 {len(file_list)} 个待处理文件。")

# 检查两个目录找到的文件数量是否一致（重要！）
if len(set([len(v) for v in files_to_rename.values()])) > 1:
    print("错误：两个目录中匹配到的文件数量不一致，请检查原始文件！")
    exit()

# 第二步：执行批量重命名
new_index = start_num
for idx in range(len(files_to_rename[sub_dirs[0]])):  # 以第一个目录的长度为基准
    for sub_dir in sub_dirs:
        old_name = files_to_rename[sub_dir][idx]
        old_path = os.path.join(base_dir, sub_dir, old_name)

        # 生成新文件名，格式为 fusc.0001.png
        new_name = f"fusc.{new_index:04d}.png"
        new_path = os.path.join(base_dir, sub_dir, new_name)

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"重命名成功：{os.path.join(sub_dir, old_name)} -> {os.path.join(sub_dir, new_name)}")
        except Exception as e:
            print(f"重命名失败 {old_path}: {e}")

    # 每处理完一对（图片+掩码），编号加1
    new_index += 1

print("批量重命名完成！")