tic
% 获取摄像头信息
info = imaqhwinfo;
win_info = imaqhwinfo('winvideo');
pixel = 'RGB24_512x384';

% 创建视频输入对象, 请自行分辨出参考相机和信号相机
inputCamera1 = videoinput('winvideo', 1, pixel); 
inputCamera2 = videoinput('winvideo', 2, pixel);

% 配置相机1参数
set(inputCamera1,'FramesPerTrigger', 1);
set(inputCamera1,'TriggerRepeat', Inf);
src1 = getselectedsource(inputCamera1);
src1.Brightness = 12000;
src1.Contrast = 100;
src1.Gain = 10;

% 配置相机2参数
set(inputCamera2,'FramesPerTrigger', 1);
set(inputCamera2,'TriggerRepeat', Inf);
src2 = getselectedsource(inputCamera2);
src2.Brightness = 12000;
src2.Contrast = 100;
src2.Gain = 10;

% 设置手动触发模式
triggerconfig(inputCamera1, 'manual');
triggerconfig(inputCamera2, 'manual');

% 创建保存目录
outputDir1 = 'Camera1_Photos';
outputDir2 = 'Camera2_Photos';
if ~exist(outputDir1, 'dir')
    mkdir(outputDir1);
end
if ~exist(outputDir2, 'dir')
    mkdir(outputDir2);
end

% 启动摄像头
start(inputCamera1);
start(inputCamera2);

% 同步拍摄循环
for frameNum = 1:100
    % 同时触发两个摄像头
    trigger(inputCamera1);
    trigger(inputCamera2);
    
    % 获取图像数据
    frame1 = getdata(inputCamera1, 1);
    frame2 = getdata(inputCamera2, 1);
    
    % 保存图像文件
    imwrite(frame1, fullfile(outputDir1, [num2str(frameNum) '.jpg']), 'Quality', 100);
    imwrite(frame2, fullfile(outputDir2, [num2str(frameNum) '.jpg']), 'Quality', 100);
    
    % 显示进度
    disp(['已拍摄第 ' num2str(frameNum) ' 张照片']);
end

% 清理资源
stop(inputCamera1);
stop(inputCamera2);
delete(inputCamera1);
delete(inputCamera2);
clear inputCamera1 inputCamera2;

toc
disp('拍摄完成！所有照片已保存至相应目录。');
