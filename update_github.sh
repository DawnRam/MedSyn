#!/bin/bash

# 1. 先将不需要跟踪的文件夹加入 .gitignore
echo -e "\ncheckpoints/\ncifar10_data/\nlogs/\nwandb/" >> .gitignore

# 2. 添加所有更改（排除 .gitignore 里已忽略的内容）
git add .

# 3. 显示状态，便于确认
git status

# 4. 获取提交信息
read -p "请输入提交信息（默认: update project）: " msg
msg=${msg:-update project}

# 5. 提交
git commit -m "$msg"

# 6. 推送
git push

echo "✅ 已完成推送，已排除指定文件夹。"
