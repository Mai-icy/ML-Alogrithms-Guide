from pathlib import Path

BASE = Path("../")

path1 = BASE / "ML_Alogrithms_Guide" / "2_Supervised_Learning"
path2 = BASE / "ML_Alogrithms_Guide" / "3_Unsupervised_Learning"


def split_file_in_dir(path: Path):
    for item in path.iterdir():
        if item.is_dir():
            split_file(item / "README.md")


def split_file(file: Path):
    text = file.read_text(encoding="utf-8")
    lines = list(text.split("\n"))

    index_list = []
    for index, line in enumerate(lines):
        if line.startswith("## 概述"):
            index_list.append(index)
        if line.startswith("## 算法说明"):
            index_list.append(index)
        if line.startswith("## 示例代码"):
            index_list.append(index)
        if line.startswith("## 详细说明"):
            index_list.append(index)

    base_path = file.parent
    overview_path = base_path / "overview.md"
    description_path = base_path / "description.md"
    example_path = base_path / "example.md"
    detailed_path = base_path / "detailed.md"

    overview_path.write_text("\n".join(lines[index_list[0]:index_list[1]]))
    description_path.write_text("\n".join(lines[index_list[1]:index_list[2]]))
    example_path.write_text("\n".join(lines[index_list[2]:index_list[3]]))
    detailed_path.write_text("\n".join(lines[index_list[3]:]))


if __name__ == '__main__':
    test_file = path1 / "1_Linear_Regression" / "README.md"
    split_file_in_dir(path1)
    split_file_in_dir(path2)
