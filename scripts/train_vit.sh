#!/bin/bash
# ==============================================
# ViT 模型训练一键管理脚本
# ==============================================
# 基础配置（与 vit_config.py 严格对齐）
TRAIN_SCRIPT="train.py"
BASE_LOG_DIR="train_results/logs"
MODEL_TYPE="vit"

# 训练默认参数（与 vit_config.py 对齐）
BATCH_SIZE=64
LR=0.0005
EPOCHS=200

# 自动生成日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BASE_LOG_DIR}/${MODEL_TYPE}_${TIMESTAMP}.log"
# 进程关键词（仅匹配ViT训练进程）
PROCESS_KEYWORD="python.*${TRAIN_SCRIPT}.*--model ${MODEL_TYPE}"

# 创建日志目录
mkdir -p "${BASE_LOG_DIR}"

# 检查进程
check_process() {
    PID=$(pgrep -f "${PROCESS_KEYWORD}")
    [ -n "$PID" ] && return 0 || return 1
}

# 启动训练
start_train() {
    if check_process; then
        echo "❌ ViT训练进程已运行(PID: $PID) 请先停止"
        exit 1
    fi

    TRAIN_CMD="python ${TRAIN_SCRIPT} \
        --model ${MODEL_TYPE} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        ${@:2}"

    echo "🚀 ViT训练启动命令: ${TRAIN_CMD}"
    echo "📝 日志文件: ${LOG_FILE}"
    nohup ${TRAIN_CMD} > ${LOG_FILE} 2>&1 &
    PID=$!
    echo "✅ ViT训练启动成功(PID: ${PID})"
    echo "🔍 查看日志: ./scripts/train_vit.sh log"
    echo "⏹️  停止训练: ./scripts/train_vit.sh stop"
}

# 查看日志
view_log() {
    LATEST_LOG=$(ls -t ${BASE_LOG_DIR}/${MODEL_TYPE}*.log 2>/dev/null | head -n1)
    if [ -z "${LATEST_LOG}" ]; then
        echo "❌ 无ViT训练日志"
        exit 1
    fi
    echo "📜 ViT实时日志(${LATEST_LOG}) (Ctrl+C退出)"
    echo "------------------------------------------------"
    tail -f ${LATEST_LOG}
}

# 停止训练
stop_train() {
    if ! check_process; then
        echo "⚠️  无运行中的ViT训练进程"
        exit 1
    fi
    echo "⏹️  停止ViT训练进程(PID: ${PID})..."
    kill ${PID} && sleep 2
    if check_process; then
        echo "⚠️  进程未停止，强制终止..."
        kill -9 ${PID}
    fi
    echo "✅ ViT训练已停止，续训: ./scripts/train_vit.sh start --resume"
}

# 查看状态
show_status() {
    if check_process; then
        LATEST_LOG=$(ls -t ${BASE_LOG_DIR}/${MODEL_TYPE}*.log 2>/dev/null | head -n1)
        echo "🟢 ViT训练中(PID: ${PID})"
        echo "📝 最新日志: ${LATEST_LOG}"
        echo "📌 最新10行日志:"
        echo "------------------------------------------------"
        tail -n 10 ${LATEST_LOG}
        echo "------------------------------------------------"
    else
        echo "🔴 ViT训练未运行"
        LATEST_LOG=$(ls -t ${BASE_LOG_DIR}/${MODEL_TYPE}*.log 2>/dev/null | head -n1)
        [ -n "${LATEST_LOG}" ] && echo "ℹ️  最新历史日志: ${LATEST_LOG}"
    fi
}

# 主入口
case "$1" in
    start) start_train "$@" ;;
    log) view_log ;;
    stop) stop_train ;;
    status) show_status ;;
    *)
        echo "用法: ./scripts/train_vit.sh [命令] [参数]"
        echo "命令："
        echo "  start   启动训练 (例: start --resume --batch-size 128)"
        echo "  log     查看实时日志"
        echo "  stop    停止训练"
        echo "  status  查看训练状态"
        ;;
esac
