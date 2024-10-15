from sklearn.model_selection import KFold
import json
import os


def write_to_file(result, output_file):
    # 获取文件所在的目录路径
    output_dir = os.path.dirname(output_file)
    # 如果目录不存在，则创建目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(output_file)
    with open(output_file, "a") as f:
        f.write(json.dumps(result) + "\n")


# random_state 是用来设置随机数种子的参数。通过设置这个种子，确保每次运行代码时，生成的随机结果是一致的
kf = KFold(n_splits=5, shuffle=True, random_state=42)
idx = 0


file_path = "Machine_learing/Data_process/5fold/HumanEval.jsonl"
data = list(map(json.loads, open(file_path)))

for train_index, test_index in kf.split(data):
    for train_per_index in train_index:
        train = data[train_per_index]
        write_to_file(
            train,
            f"Machine_learing/Data_process/5fold/Final_result_5fold/HumanEval_nine_train_{idx}.jsonl",
        )
    for test_per_index in test_index:
        test = data[test_per_index]
        write_to_file(
            test,
            f"Machine_learing/Data_process/5fold/Final_result_5fold/HumanEval_nine_test_{idx}.jsonl",
        )
    idx += 1
