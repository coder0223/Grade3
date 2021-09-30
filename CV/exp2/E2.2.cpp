#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap(0);//打开本地摄像头
	if (!cap.isOpened())
	{
		cout << "Open failed!\n";
		return -1;
	}
	int frame_num = 100;//设定帧数
	string outputVideoPath = "..\\vedio_out.avi";//定义录制的视频输出路径

	cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));//获取图像的宽和高并作为输出视频的宽高
	VideoWriter outputvideo;
	outputvideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
	//第一个参数表明输出路径，第二个参数为视频编码方式，第三个参数为视频流的帧速率，第四个参数为帧的尺寸
	Mat frame;
	while (cap.isOpened() && frame_num>0)
	{
		cap >> frame;//捕获每一帧
		if (frame.empty()) break;
		Mat temp = Mat::zeros(frame.size(), frame.type());//对每一帧都变形处理，达到哈哈镜效果
		double x1, y1;
		double x_t, y_t;
		int W = frame.rows;
		int H = frame.cols;
		for (int x = 0; x < W; x++)
		{
			for (int y = 0; y < H; y++)
			{
				x1 = 2.0 * x / W - 1;
				y1 = 2.0 * y / H - 1;
				double r = sqrt(x1 * x1 + y1 * y1);
				if (r >= 1)
				{
					x_t = x1;
					y_t = y1;
					for (int c = 0; c < 3; c++)
					{
						int x_map = (x_t + 1) / 2.0 * W;
						int y_map = (y_t + 1) / 2.0 * H;
						temp.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(frame.at<Vec3b>(x_map, y_map)[c]);
					}
				}
				else
				{
					double theta = (1 - r) * (1 - r);
					x_t = cos(theta) * x1 - sin(theta) * y1;
					y_t = sin(theta) * x1 + cos(theta) * y1;
					for (int c = 0; c < 3; c++)
					{
						int x_map = (x_t + 1) / 2.0 * W;
						int y_map = (y_t + 1) / 2.0 * H;
						temp.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(frame.at<Vec3b>(x_map, y_map)[c]);
					}
				}
			}
		}
		outputvideo << temp;//写入一帧视频
		frame_num--;//帧数-1
		imshow("img", temp);//显示每一帧
		waitKey(10);
		if (char(waitKey(1)) == 'q') break;
	}
	outputvideo.release();//
	system("pause");
}