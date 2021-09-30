#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap(0);//�򿪱�������ͷ
	if (!cap.isOpened())
	{
		cout << "Open failed!\n";
		return -1;
	}
	int frame_num = 100;//�趨֡��
	string outputVideoPath = "..\\vedio_out.avi";//����¼�Ƶ���Ƶ���·��

	cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));//��ȡͼ��Ŀ�͸߲���Ϊ�����Ƶ�Ŀ��
	VideoWriter outputvideo;
	outputvideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
	//��һ�������������·�����ڶ�������Ϊ��Ƶ���뷽ʽ������������Ϊ��Ƶ����֡���ʣ����ĸ�����Ϊ֡�ĳߴ�
	Mat frame;
	while (cap.isOpened() && frame_num>0)
	{
		cap >> frame;//����ÿһ֡
		if (frame.empty()) break;
		Mat temp = Mat::zeros(frame.size(), frame.type());//��ÿһ֡�����δ����ﵽ������Ч��
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
		outputvideo << temp;//д��һ֡��Ƶ
		frame_num--;//֡��-1
		imshow("img", temp);//��ʾÿһ֡
		waitKey(10);
		if (char(waitKey(1)) == 'q') break;
	}
	outputvideo.release();//
	system("pause");
}