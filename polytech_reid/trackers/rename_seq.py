import os


dataset_path = r'D:\CodeProjects\PythonProjects\polytech_reid\datasets\person_seq_2_1'
start_index = 2775

for img_fn in sorted(os.listdir(os.path.join(dataset_path, 'img1')), key=lambda x: -int(os.path.splitext(x)[0])):
    name, ext = os.path.splitext(img_fn)
    new_name = str(int(name) + 1 - start_index)
    new_img_fn = new_name + ext
    os.rename(os.path.join(dataset_path, 'img1', img_fn), os.path.join(dataset_path, 'img1', new_img_fn))
    print(img_fn, new_img_fn)

# gt_path = os.path.join(dataset_path, 'gt', 'gt.txt')
# with open(gt_path, 'r') as f:
#     text = f.read()

# lines = text.split('\n')
# new_lines = []
# for line in lines:
#     if line == '':
#         continue

#     elems = line.split(',')
#     elems[0] = str(int(elems[0]) + 1 - start_index)
#     new_line = ','.join(elems)
#     print(line, new_line)
#     new_lines.append(new_line)

# new_text = '\n'.join(new_lines)
# with open(gt_path, 'w') as f:
#     f.write(new_text)
