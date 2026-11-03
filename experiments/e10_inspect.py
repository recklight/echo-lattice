"""List the layout of an HDF5 file: groups, datasets with shape, dtype and attributes. No data values.

    python -I e10_inspect.py <file.hdf5> [max_depth] [max_children]

External and soft links are reported, never followed.
"""

import sys

import h5py


def walk(group, depth, max_depth, max_children, indent=""):
    names = list(group.keys())
    print(f"{indent}{group.name}  ({len(names)} members)  attrs={dict(group.attrs)}")
    if depth >= max_depth:
        return
    for name in names[:max_children]:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            print(f"{indent}  {name}: {type(link).__name__} (not followed)")
            continue
        item = group[name]
        if isinstance(item, h5py.Dataset):
            print(f"{indent}  {name}: dataset shape={item.shape} dtype={item.dtype} "
                  f"chunks={item.chunks} compression={item.compression} attrs={dict(item.attrs)}")
        else:
            walk(item, depth + 1, max_depth, max_children, indent + "  ")
    if len(names) > max_children:
        print(f"{indent}  ... {len(names) - max_children} more: {names[max_children:max_children + 20]}")


def main():
    path = sys.argv[1]
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    max_children = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    with h5py.File(path, "r") as f:
        print("root attrs:", dict(f.attrs))
        walk(f, 0, max_depth, max_children)


if __name__ == "__main__":
    main()
