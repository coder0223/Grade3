#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

const char* video_path = "E:/1ComputerVision/exp/exp4/demo_test.mp4";
Mat img, temp, outputImg;
Point begin_point;
Point end_point;//矩形框的起点和终点
bool left_flag=false;//暂停播放进行标记

int c[] = { 0,1 };
int hist_size[] = { 30,32 };
float rangeH[] = { 0,180 };
float rangeS[] = { 0,256 };
const float* ranges[] = { rangeH,rangeS };

void draw_hist(const Mat image)//绘制直方图
{
	int img_num = 1;//表示图片数量
	int c[3] = { 0,1,2 };//图像的通道数量
	Mat out_red, out_green, out_blue;//输出三维度的直方图
	int dims = 1;//维度
	int hist_size[3] = { 256,256,256 };//存放每个维度的直方图尺寸
	float hist_range[2] = { 0,255 };//每一维直方图的取值范围
	const float* ranges[3] = { hist_range,hist_range,hist_range };//值范围的指针
	bool uniform = true;//表示是否均匀
	bool accum = false;//表示是否积累
	//调用calHist函数计算每个通道下的直方图
	cv::calcHist(&image, img_num, &c[0], cv::Mat(), out_red, dims, &hist_size[0], &ranges[0], uniform, accum);
	cv::calcHist(&image, img_num, &c[1], cv::Mat(), out_green, dims, &hist_size[1], &ranges[1], uniform, accum);
	cv::calcHist(&image, img_num, &c[2], cv::Mat(), out_blue, dims, &hist_size[2], &ranges[2], uniform, accum);

	int scale = 1;
	Mat histRect(hist_size[0], hist_size[0] * scale * 3, CV_8UC3, Scalar(0, 0, 0));
	double max_val[3] = { 0,0,0 };
	double min_val[3] = { 0,0,0 };//找到每个通道的最值
	minMaxLoc(out_red, &min_val[0], &max_val[0], NULL, NULL);
	minMaxLoc(out_green, &min_val[1], &max_val[1], NULL, NULL);
	minMaxLoc(out_blue, &min_val[2], &max_val[2], NULL, NULL);

	//坐标缩放比例
	double rate_red = (hist_size[0] / max_val[0]) * 0.9;
	double rate_green = (hist_size[1] / max_val[1]) * 0.9;
	double rate_blue = (hist_size[2] / max_val[2]) * 0.9;
	
	for (int i = 0; i < 256; i++)
	{
		float red_val = out_red.at<float>(i);
		float green_val = out_green.at<float>(i);
		float blue_val = out_blue.at<float>(i);
		line(histRect, Point(i * scale, hist_size[0]), Point(i * scale, hist_size[0] - red_val * rate_red), Scalar(0, 0, 255));
		line(histRect, Point((i+256) * scale, hist_size[0]), Point((i+256) * scale, hist_size[0] - green_val * rate_green), Scalar(0, 255, 0));
		line(histRect, Point((i+512) * scale, hist_size[0]), Point((i+512) * scale, hist_size[0] - blue_val * rate_blue), Scalar(255, 0, 0));
	}
	imshow("hist-rect", histRect);
}

double compare_hist(const Mat srcHist, Mat tar_img)//比较直方图的相似度
{
	Mat HSVimg;//在HSV空间中进行比较操作
	cvtColor(tar_img, HSVimg, CV_BGR2HSV);//将BGR空间转换为HSV空间
	Mat compHist;
	calcHist(&HSVimg, 1, c, Mat(), compHist, 2, hist_size, ranges, true, false);//计算直方图
	normalize(compHist, compHist, 0, 1, NORM_MINMAX);//归一化操作
	//使用compareHist函数计算两个直方图的相似度
	int method = 3;//表示采用巴氏距离比较两个直方图的相似度
	double res = compareHist(srcHist, compHist, method);
	return res;
}

void target_trace(const Mat srcHist)
{
	int width = abs(end_point.x - begin_point.x);
	int height = abs(end_point.y - begin_point.y);
	//目标搜索区域设定为原区域的周围且面积为原来的三倍
	int x1 = begin_point.x - width;
	int x2 = begin_point.x + width;
	int y1 = begin_point.y - height;
	int y2 = begin_point.y + height;
	if (x1 < 0) x1 = 0;
	if (y1 < 0) y1 = 0;//越界检查

	Point start, end;
	Point get_start(0, 0);
	Point get_end(0, 0);

	VideoCapture video(video_path);
	if (!video.isOpened())
	{
		cout << "video open failed!";
		return;
	}
	double fps = video.get(CV_CAP_PROP_FPS);//获取视频帧率
	double pause = 1000 / fps;
	int w = static_cast<int>(video.get(CV_CAP_PROP_FRAME_WIDTH));
	int h = static_cast<int>(video.get(CV_CAP_PROP_FRAME_HEIGHT));
	Size S(w, h);
	while (1)
	{
		video >> img;
		if (img.empty() || waitKey(pause) == 27)
		{
			break;
		}
		double threshold = 1.0;//直方图对比的相似度阈值
		for (int y = y1; y <= y2; y+=10)
		{
			for (start.x = x1, start.y = y; start.x <= x2; start.x += 10)
			{
				if ((start.x + width) < img.cols)
				{
					end.x = start.x + width;
				}
				else
				{
					end.x = img.cols - 1;
				}
				if ((start.y + height) < img.rows)
				{
					end.y = start.y + height;
				}
				else
				{
					end.y = img.rows - 1;
				}

				Mat compImg;
				compImg = img(Rect(start, end));
				double ans = compare_hist(srcHist, compImg);
				if (ans < threshold)
				{
					get_start = start;
					get_end = end;
					threshold = ans;
				}
			}
		}
		if (threshold < 0.15)
		{
			x1 = get_start.x - width;
			x2 = get_start.x + width;
			y1 = get_start.y - height;
			y2 = get_start.y + height;
			if (x1 < 0) x1 = 0;
			if (y1 < 0) y1 = 0;
		}
		if (threshold < 0.5)
		{
			rectangle(img, get_start, get_end, Scalar(0, 0, 255), 2);
		}
	}
	video.release();
}

void mouse(int action, int x, int y, int flag, void* ustc)
{
	if (action == CV_EVENT_LBUTTONDOWN)//左键按下
	{
		left_flag = true;
		begin_point = Point(x, y);//记录矩形框的起点坐标
		end_point = begin_point;
	}
	if (action == CV_EVENT_MOUSEMOVE && left_flag)//如果左键已按下并移动鼠标
	{
		temp = img.clone();
		end_point = Point(x, y);//记录矩形框的终点坐标
		if (end_point != begin_point)
		{
			rectangle(temp, begin_point, end_point, Scalar(255, 0, 0), 2);
		}
		imshow("Video", temp);
	}
	if (action == CV_EVENT_LBUTTONUP)//左键抬起，目标标记完成，可开始追踪
	{
		left_flag = false;
		outputImg = img(Rect(begin_point, end_point));
	}
}

int main()
{
	VideoCapture cap(video_path);
	if (!cap.isOpened())
	{
		cout << "video open failed!";
		return -1;
	}
	double fps = cap.get(CV_CAP_PROP_FPS);//获取视频帧率
	double pause = 1000 / fps;
	namedWindow("Video",WINDOW_AUTOSIZE);
	setMouseCallback("Video", mouse);
	while (true)
	{
		if (left_flag == false)
		{
			cap >> img;
		}
		if (img.empty() || waitKey(pause) == 27)
		{
			break;
		}
		if (begin_point != end_point && left_flag == false)//移动过鼠标且左键已抬起，绘制矩形框
		{
			rectangle(img, begin_point, end_point, Scalar(255, 0, 0), 2);
		}
		imshow("Video", img);
	}
	cap.release();
	//imshow("Traget", outputImg);
	draw_hist(outputImg);
	Mat outputImg_HSV;
	cvtColor(outputImg, outputImg_HSV, CV_RGB2HSV);//将RGB颜色空间转化为HSV颜色空间
	Mat srcHist;
	calcHist(&outputImg_HSV, 1, c, Mat(), srcHist, 2, hist_size, ranges, true, false);
	normalize(srcHist, srcHist, 0, 1, NORM_MINMAX);//归一化操作
	target_trace(srcHist);
	waitKey(0);
	return 0;
}