#!/usr/bin/env python3
import base64
import zlib
import json
import hashlib
import re
import sys
import shutil
import datetime
from pathlib import Path

# 云端修改与原始 ZIP 的差异，压缩存储；不需要联网。
DATA = """
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
5O9pI42EcPEY7G//H5pckug=
"""

root = (
    Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) > 1
    else Path.home() / "robocon/robocon3"
)
changes = json.loads(zlib.decompress(base64.b64decode(DATA)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_patch(old, patch):
    source = old.decode("utf-8").splitlines(True)
    lines = patch.splitlines(True)
    result = []
    pos = 0
    i = 2

    while i < len(lines):
        match = re.match(
            r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", lines[i]
        )
        if not match:
            raise RuntimeError("补丁格式错误")

        start = max(0, int(match[1]) - 1)
        if start < pos:
            raise RuntimeError("补丁顺序错误")

        result.extend(source[pos:start])
        pos = start
        i += 1

        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            i += 1

            if line.startswith("\\"):
                continue

            if line[0] in " -":
                if pos >= len(source) or source[pos] != line[1:]:
                    raise RuntimeError("补丁上下文不匹配")
                if line[0] == " ":
                    result.append(source[pos])
                pos += 1
            elif line[0] == "+":
                result.append(line[1:])
            else:
                raise RuntimeError("补丁行格式错误")

    result.extend(source[pos:])
    return "".join(result).encode("utf-8")


if not root.is_dir():
    sys.exit("项目目录不存在：" + str(root))

# 所有校验完成后才开始备份和写入。
prepared = {}
for name, item in changes.items():
    path = root / name
    if root not in path.resolve().parents:
        sys.exit("不允许的文件路径")

    old = path.read_bytes() if path.exists() else b""

    if path.exists() and digest(old) == item["new"]:
        continue

    if (
        item["old"] is None and path.exists()
    ) or (
        item["old"] is not None
        and (not path.exists() or digest(old) != item["old"])
    ):
        sys.exit(
            "停止，文件与上传的原版不同，未修改任何文件：" + name
        )

    new = apply_patch(old, item["patch"])
    if digest(new) != item["new"]:
        sys.exit("补丁校验失败：" + name)
    prepared[name] = new

if not prepared:
    print("已是修改后的版本，无需重复更新。")
    sys.exit(0)

backup = root.parent / (
    root.name
    + "_backup_"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
backup.mkdir()

for name in prepared:
    path = root / name
    if path.exists():
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

(backup / "新增文件清单.txt").write_text(
    "\n".join(name for name in prepared if not (root / name).exists()),
    encoding="utf-8",
)

try:
    for name, new in prepared.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(new)
except Exception:
    for name in prepared:
        path = root / name
        saved = backup / name
        if saved.exists():
            shutil.copy2(saved, path)
        elif path.exists():
            path.unlink()
    raise

print("更新完成：", root)
print("原文件备份：", backup)
print("修改文件数：", len(prepared))
print("请阅读项目内的 修改说明.md")