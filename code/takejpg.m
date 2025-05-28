tic
% ��ȡ����ͷ��Ϣ
info = imaqhwinfo;
win_info = imaqhwinfo('winvideo');
pixel = 'RGB24_512x384';

% ������Ƶ�������, �����зֱ���ο�������ź����
inputCamera1 = videoinput('winvideo', 1, pixel); 
inputCamera2 = videoinput('winvideo', 2, pixel);

% �������1����
set(inputCamera1,'FramesPerTrigger', 1);
set(inputCamera1,'TriggerRepeat', Inf);
src1 = getselectedsource(inputCamera1);
src1.Brightness = 12000;
src1.Contrast = 100;
src1.Gain = 10;

% �������2����
set(inputCamera2,'FramesPerTrigger', 1);
set(inputCamera2,'TriggerRepeat', Inf);
src2 = getselectedsource(inputCamera2);
src2.Brightness = 12000;
src2.Contrast = 100;
src2.Gain = 10;

% �����ֶ�����ģʽ
triggerconfig(inputCamera1, 'manual');
triggerconfig(inputCamera2, 'manual');

% ��������Ŀ¼
outputDir1 = 'Camera1_Photos';
outputDir2 = 'Camera2_Photos';
if ~exist(outputDir1, 'dir')
    mkdir(outputDir1);
end
if ~exist(outputDir2, 'dir')
    mkdir(outputDir2);
end

% ��������ͷ
start(inputCamera1);
start(inputCamera2);

% ͬ������ѭ��
for frameNum = 1:100
    % ͬʱ������������ͷ
    trigger(inputCamera1);
    trigger(inputCamera2);
    
    % ��ȡͼ������
    frame1 = getdata(inputCamera1, 1);
    frame2 = getdata(inputCamera2, 1);
    
    % ����ͼ���ļ�
    imwrite(frame1, fullfile(outputDir1, [num2str(frameNum) '.jpg']), 'Quality', 100);
    imwrite(frame2, fullfile(outputDir2, [num2str(frameNum) '.jpg']), 'Quality', 100);
    
    % ��ʾ����
    disp(['������� ' num2str(frameNum) ' ����Ƭ']);
end

% ������Դ
stop(inputCamera1);
stop(inputCamera2);
delete(inputCamera1);
delete(inputCamera2);
clear inputCamera1 inputCamera2;

toc
disp('������ɣ�������Ƭ�ѱ�������ӦĿ¼��');
