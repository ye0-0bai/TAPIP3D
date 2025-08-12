import os
import subprocess
import multiprocessing
from multiprocessing import Process, Queue, current_process
from tqdm import tqdm
import argparse
import time

def process_data(data_path, gpu_id):
    """
    处理单个数据的函数
    :param data_path: 要处理的数据路径
    :param gpu_id: 使用的GPU ID
    """
    try:
        # 设置环境变量，指定使用的GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id%8)
        
        # 这里替换成你实际的命令
        command = f"python /data/repository/TAPIP3D/process_calvin.py --input_path {data_path}"
        
        # 执行命令
        with open(f'/data/repository/TAPIP3D/parallel_processor_logs/gpu_{gpu_id}.txt', 'a') as f:
            subprocess.run(command, shell=True, check=True, env=env, stdout=f, stderr=f)
        return True, data_path, gpu_id
    except subprocess.CalledProcessError as e:
        return False, data_path, gpu_id

def worker(task_queue, result_queue, gpu_id):
    """
    工作进程函数
    :param task_queue: 任务队列
    :param result_queue: 结果队列
    :param gpu_id: 分配的GPU ID
    """
    while True:
        # 从队列获取任务
        data_path = task_queue.get()
        
        # 如果收到None，表示任务结束
        if data_path is None:
            break
        
        # 处理数据
        success, processed_path, used_gpu = process_data(data_path, gpu_id)
        
        # 将结果放入结果队列
        result_queue.put((success, processed_path, used_gpu))

def progress_monitor(result_queue, total_tasks):
    """
    进度监控函数
    :param result_queue: 结果队列
    :param total_tasks: 总任务数
    """
    success_count = 0
    failure_count = 0
    
    with tqdm(total=total_tasks, desc="处理进度", ncols=100) as pbar:
        while success_count + failure_count < total_tasks:
            # 从结果队列获取结果
            success, data_path, gpu_id = result_queue.get()
            
            if success:
                success_count += 1
                # print(f"成功处理: {data_path} (GPU {gpu_id})")
            else:
                failure_count += 1
                print(f"\n处理失败: {data_path} (GPU {gpu_id})")
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "失败": failure_count})

def main(data_list, num_gpus=8):
    """
    主函数
    :param data_list: 要处理的数据路径列表
    :param num_gpus: 可用的GPU数量
    """
    # 创建任务队列和结果队列
    task_queue = Queue()
    result_queue = Queue()
    
    # 将任务放入队列
    for data_path in data_list:
        task_queue.put(data_path)
    
    # 添加结束标志（每个worker一个None）
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 创建worker进程
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(task_queue, result_queue, gpu_id))
        p.start()
        workers.append(p)
    
    # 创建并启动进度监控进程
    monitor = Process(target=progress_monitor, args=(result_queue, len(data_list)))
    monitor.start()
    
    # 等待所有worker完成
    for p in workers:
        p.join()
    
    # 等待进度监控完成
    monitor.join()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="并行数据处理脚本")
    parser.add_argument("--num_gpus", type=int, default=8, help="可用的GPU数量")
    
    args = parser.parse_args()
    
    os.makedirs('/data/repository/TAPIP3D/parallel_processor_logs')
        
    data_list = [os.path.join('/mnt/nas_24/wangwq/datasets/calvin/refactored/task_ABC_D/validation/static_camera_observation', file_name) for file_name in os.listdir('/mnt/nas_24/wangwq/datasets/calvin/refactored/task_ABC_D/validation/static_camera_observation')]

    print(f"开始处理 {len(data_list)} 个数据文件，使用 {args.num_gpus} 张GPU...")
    main(data_list, args.num_gpus)
    print(f"所有数据处理完成!")
