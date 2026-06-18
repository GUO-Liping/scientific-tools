# 该程序用于4通道OBS数据整理：默认为7天数据，1个通道为1个文件

import os
import glob
from obspy import read, Stream, UTCDateTime

# OBS监测设备-4个通道
EXTENSIONS = [".BHZ", ".BHN", ".BHE", ".HYD"]

# 合并时间跨度
MERGE_DAYS = 7  

def merge_seismic_data():
    for ext in EXTENSIONS:
        print(f"\n处理通道: {ext} ")
        
        # 1. 获取所有当前通道文件
        files = glob.glob(f"*{ext}")
        
        if not files:
            print(f"未在当前目录下找到任何 *{ext} 文件")
            continue

        print(f"找到 {len(files)} 个 {ext} 文件，解析时间标签并排序...")

        # 2. 快速读取文件头并排序
        file_times = []
        for f in files:
            try:
                st_head = read(f, headonly=True)
                if len(st_head) > 0:
                    start = st_head[0].stats.starttime
                    end = st_head[-1].stats.endtime
                    file_times.append((start, end, f))
            except Exception as e:
                print(f"无法读取文件头 {f}: {e}")

        if not file_times:
            continue

        # 按开始时间严格排序
        file_times.sort(key=lambda x: x[0])
        
        # 3. 按照指定天数将文件列表分成多个时间组
        global_start = file_times[0][0]
        time_chunks = []
        
        current_chunk_start = global_start
        current_chunk_end = current_chunk_start + (MERGE_DAYS * 86400)
        current_chunk_files = []
        
        for start, end, f in file_times:
            while start >= current_chunk_end:
                if current_chunk_files:
                    time_chunks.append((current_chunk_start, current_chunk_end, current_chunk_files))
                current_chunk_start = current_chunk_end
                current_chunk_end = current_chunk_start + (MERGE_DAYS * 86400)
                current_chunk_files = []
            
            current_chunk_files.append((start, end, f))
            
        if current_chunk_files:
            time_chunks.append((current_chunk_start, current_chunk_end, current_chunk_files))

        print(f"将所有数据分为 {len(time_chunks)} 组（每组 {MERGE_DAYS} 天）")

        # 4. 逐个时间组进行合并、保存和绘图
        for chunk_idx, (chunk_start, chunk_end, chunk_files) in enumerate(time_chunks, 1):
            
            actual_start_str = chunk_files[0][0].strftime("%Y%m%dT%H%M%S")
            actual_end_str = chunk_files[-1][1].strftime("%Y%m%dT%H%M%S")
            
            output_filename = f"{actual_start_str}-{actual_end_str}{ext}"
            
            print(f"\n[组 {chunk_idx}/{len(time_chunks)}] 正在合并时间段 {actual_start_str} 至 {actual_end_str}，共 {len(chunk_files)} 个文件...")
            
            main_stream = Stream()
            
            for _, _, f in chunk_files:
                try:
                    main_stream += read(f)
                except Exception as e:
                    print(f"读取波形数据失败 {f}: {e}")
            
            if len(main_stream) == 0:
                print("当前时间组内无有效数据，跳过。")
                continue
                
            main_stream.merge(method=1, fill_value=0)

            # 保存合并后的文件
            print(f"正在保存合并文件至: {output_filename}")
            main_stream.write(output_filename, format="MSEED")

            # 5. 修正后的高效经典绘图（横轴：时间，纵轴：幅值）
            img_filename = f"{actual_start_str}-{actual_end_str}_{ext.strip('.')}.png"
            
            try:
                # 强制采用标准波形图，并限定单通道最大绘制点数（例如10万点），防止大数据量卡死
                # ObsPy 会在保持波形特征的前提下自动下采样，画出来的图依旧是：横坐标时间，纵坐标幅值
                main_stream.plot(
                    outfile=img_filename, 
                    size=(1500, 500),
                    color="black",
                    linewidth=0.5,
                    max_points=100000  # 核心效率参数：限制绘图点数，极大加快超大数据文件的画图速度
                )
                print(f"预览图已保存至: {img_filename}")
            except Exception as e:
                print(f"绘图遇到错误: {e}")
            
            # 及时清理内存
            del main_stream

    print("\n 所有通道操作完成！ ")

if __name__ == "__main__":
    merge_seismic_data()