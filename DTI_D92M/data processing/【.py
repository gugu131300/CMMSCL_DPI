# 定义数字范围和折数
num_range = list(range(24820))
folds = 6

# 计算每一折的数字数量
fold_size = len(num_range) // folds
remainder = len(num_range) % folds

# 初始化折数列表
fold_lists = []

# 将数字均匀分配到每一折
start = 0
for i in range(folds):
    fold_end = start + fold_size + (1 if i < remainder else 0)
    fold_lists.append(num_range[start:fold_end])
    start = fold_end

# 格式化输出
formatted_folds = [f"[{','.join(map(str, fold))}]" for fold in fold_lists]

# 输出结果
for fold in formatted_folds:
    print(fold)