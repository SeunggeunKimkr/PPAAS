#!/usr/bin/env python3
import os
import re

def convert_rel_to_abs(file_path):
    """
    In a given .cir file, find any .include or .lib directive whose path
    is relative (starts with ./ or ../) and replace it with its absolute path.
    """
    # Matches lines like:
    #   .include "../models/transistor.lib"
    #   .lib    ./subckt/foo.sub ff
    pattern = re.compile(
        r'^(?P<dir>\.include|\.lib)\s+'         # directive
        r'"?(?P<rel>\.?\.?/[^\s"]+)"?'          # relative path in quotes or not
        r'(?P<suffix>.*)$'                      # any trailing flags/options
    )
    file_dir = os.path.dirname(file_path)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        m = pattern.match(line.strip())
        if m:
            rel_path = m.group('rel')
            # compute the absolute path
            abs_path = os.path.normpath(os.path.join(file_dir, rel_path))
            # rebuild the line with quotes around the absolute path
            new_line = f'{m.group("dir")} "{abs_path}"{m.group("suffix")}\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)


def main():
    # Assume you run this from:
    root = os.getcwd()
    bak_dir = os.path.join(root, 'netlist')

    for dirpath, _, files in os.walk(bak_dir):
        for fn in files:
            if fn.endswith('.cir'):
                full = os.path.join(dirpath, fn)
                print(f'Converting {full}')
                convert_rel_to_abs(full)


if __name__ == "__main__":
    main()
