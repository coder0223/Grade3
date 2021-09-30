#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

Mat Bilinear(Mat img, double a, double b);

int main()
{
	Mat img = imread("E:/1ComputerVision/exp/exp2/pic.jpg");
	Mat temp = Mat::zeros(img.size(), img.type());
	namedWindow("Image",WINDOW_AUTOSIZE);
	imshow("Image",img);
	double x1, y1;
	double x_t, y_t;
	int W = img.rows;
	int H = img.cols;
	for (int x = 0; x < W; x++)
	{
		for (int y = 0; y < H; y++)
		{
			x1 = 2.0 * x / W - 1;
			y1 = 2.0 * y / H - 1;
			double r = sqrt(x1 * x1 + y1 * y1);
			if (r>=1)
			{
				x_t = x1;
				y_t = y1;
				for (int c = 0; c < 3; c++)
				{
					int x_map = (x_t + 1) / 2.0 * W;
					int y_map = (y_t + 1) / 2.0 * H;
					temp.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(img.at<Vec3b>(x_map, y_map)[c]);
				}
			}
			else
			{
				double theta = (1 - r) * (1 - r);
				x_t = cos(theta) * x1 - sin(theta) * y1;
				y_t = sin(theta) * x1 + cos(theta) * y1;
				for (int c = 0; c < 3; c++)
				{
					int x_map = (x_t+1) / 2.0 * W;
					int y_map = (y_t+1) / 2.0 * H;
					temp.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(img.at<Vec3b>(x_map,y_map)[c]);
				}
			}
		}
	}
	Mat co = Bilinear(temp, 0.8, 0.8);
	//Mat co2 = Bilinear(temp,1.2,1.2;
	namedWindow("COPY", WINDOW_AUTOSIZE);
	namedWindow("Transformate", WINDOW_AUTOSIZE);
	//namedWindow("Copy2",WINDOW_AUTOSIZE);
	//imshow("Copy2",co2);
	imshow("TransFormate", temp);
	imshow("COPY", co);
	waitKey(0);
	return 0;
}

Mat Bilinear(Mat img, double a, double b)//双线性插值函数
{
	int row = (int)(img.rows * a);
	int col = (int)(img.cols * b);
	Mat Copy(row, col, CV_8UC3);//按缩放后图片的宽高生成图片
	for (int x = 0; x < Copy.rows; x++)
	{
		for (int y = 0; y < Copy.cols; y++)
		{
			int t_x = (int)(x / a);
			int t_y = (int)(y / b);
			if (t_x >= img.rows - 1 || t_y >= img.cols - 1)//处理边缘位置的像素
			{
				for (int c = 0; c < 3; c++)
				{
					Copy.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(img.at<Vec3b>(img.rows-1, img.cols-1)[c]);
				}
			}
			else
			{
				double dx = x / a - t_x;
				double dy = y / b - t_y;//计算原图像坐标减去新图像坐标后剩下的小数部分
				for (int c = 0; c < 3; c++)
				{
					Copy.ptr(x, y)[c] = img.ptr(t_x, t_y)[c] * (1 - dx) * (1 - dy) + img.ptr(t_x, t_y + 1)[c] * (1-dx) * dy + img.ptr(t_x + 1, t_y)[c] * (1 - dy) * dx + img.ptr(t_x + 1, t_y + 1)[c] * dx * dy;
					//Copy.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(img.at<Vec3b>(t_x, t_y)[c] * (1 - dx) * (1 - dy) + img.at<Vec3b>(t_x, t_y + 1)[c] * dx * (1 - dy) + img.at<Vec3b>(t_x + 1, t_y) * (1 - dx) * dy + img.at<Vec3b>(t_x + 1, t_y + 1)[c] * dx * dy);
				}
			}
		}
	}
	return Copy;
}
