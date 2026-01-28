from obspy import read
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取地震波形数据
# PZ水文站obs设备4通道数据
pz_st1_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHE"
pz_st1_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHN"
pz_st1_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHZ"
pz_st1_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.HYD"

pz_st2_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHE"
pz_st2_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHN"
pz_st2_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHZ"
pz_st2_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.HYD"

pz_st3_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHE"
pz_st3_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHN"
pz_st3_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHZ"
pz_st3_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.HYD"

pz_st4_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHE"
pz_st4_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHN"
pz_st4_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHZ"
pz_st4_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.HYD"

pz_st5_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHE"
pz_st5_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHN"
pz_st5_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHZ"
pz_st5_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.HYD"

pz_st6_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHE"
pz_st6_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHN"
pz_st6_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHZ"
pz_st6_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.HYD"

pz_st7_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHE"
pz_st7_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHN"
pz_st7_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHZ"
pz_st7_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.HYD"


# MT水文站ob4通道数据
mt_st1_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHE"
mt_st1_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHN"
mt_st1_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHZ"
mt_st1_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.HYD"

mt_st2_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHE"
mt_st2_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHN"
mt_st2_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHZ"
mt_st2_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.HYD"

mt_st3_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHE"
mt_st3_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHN"
mt_st3_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHZ"
mt_st3_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.HYD"

mt_st4_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHE"
mt_st4_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHN"
mt_st4_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHZ"
mt_st4_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.HYD"

mt_st5_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHE"
mt_st5_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHN"
mt_st5_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHZ"
mt_st5_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.HYD"

mt_st6_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHE"
mt_st6_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHN"
mt_st6_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHZ"
mt_st6_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.HYD"

mt_st7_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHE"
mt_st7_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHN"
mt_st7_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHZ"
mt_st7_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.HYD"

pz_data_BHE = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.BHE"
pz_data_BHN = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.BHN"
pz_data_BHZ = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.BHZ"
pz_data_HYD = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.HYD"

# 选择数据
st = read(pz_data_BHN)

# 打印头部信息
print(st[0].stats)

# 绘制OBS数据
fig = st.plot(handle=True, color='k', equal_scale=False, linewidth=0.5)
fig.set_size_inches(12, 4)
fig.subplots_adjust(top=0.855, bottom=0.2, left=0.1, right=0.95, hspace=0.0, wspace=0.2)

# 设置时间刻度
ax = fig.axes[0]
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 主刻度：每12小时
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # 次刻度：每2小时
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

fig.autofmt_xdate()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Amplitude")

# 定义时间窗的开始和结束时间
start_time = st[0].stats.starttime + 5.999*24*60*60  # 起始时间：10秒
end_time = st[0].stats.starttime + 6.001*24*60*60    # 结束时间：20秒
st.trim(starttime=start_time, endtime=end_time)
st.spectrogram(log=False, title='BW.RJOB ' + str(st[0].stats.starttime))

plt.show()