import os

def print_tree(root, prefix="", level=0, max_level=3):
    if level >= max_level:
        return

    try:
        files = sorted(os.listdir(root))
    except PermissionError:
        return

    for i, name in enumerate(files):
        path = os.path.join(root, name)
        connector = "└── " if i == len(files) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if i == len(files) - 1 else "│   "
            print_tree(path, prefix + extension, level + 1, max_level)

print_tree(".")
